"""
Data utilities module for workflow analysis.
Contains functions for loading, processing, and transforming workflow data.
"""

import os
import glob
import json
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from .workflow_config import (
    WF_PARAMS, STORAGE_LIST, TARGET_TASKS, 
    TEST_CONFIGS, DEFAULT_WF, OP_DICT, MULTI_NODES
)


def transform_store_code(storage_type: str) -> int:
    """Transform storage type string to numeric code."""
    storage_mapping = {
        "ssd": 0,
        "beegfs": 1,
        "lustre": 2,
        "tmpfs": 3,
        "nfs": 4,
        "pfs": 5  # Add pfs mapping
    }
    return storage_mapping.get(storage_type, 6)


def decode_store_code(store_code: int) -> str:
    """Decode numeric storage code to string."""
    mapping = {
        0: "ssd",
        1: "beegfs",
        2: "lustre",
        3: "tmpfs",
        4: "nfs",
        5: "pfs"  # Add pfs mapping
    }
    return mapping.get(store_code, "unknown")


def bytes_to_mb(file_size: str | int | float) -> float:
    """
    Convert a file size from bytes to megabytes (MB).
    
    Parameters:
    - file_size (str, int, or float): The file size in bytes (as an int/float) 
      or a string representation with size and unit (e.g., "1024 KiB").
      
    Returns:
    - float: The file size in MB.
    """
    # If file_size is a string, parse the value and unit
    if isinstance(file_size, str):
        size_num, size_unit = file_size.split()
        size_num = float(size_num)
        
        # Convert size to MB based on the unit
        if size_unit == "Bytes" or size_unit == "B":
            return size_num / (1024 ** 2)  # Convert bytes to MB
        elif size_unit == "KiB":
            return size_num / 1024  # Convert KiB to MB
        elif size_unit == "MiB":
            return size_num  # Already in MB
        elif size_unit == "GiB":
            return size_num * 1024  # Convert GiB to MB
        elif size_unit == "TiB":
            return size_num * (1024 ** 2)  # Convert TiB to MB
        else:
            raise ValueError(f"Unknown size unit: {size_unit}")
    elif isinstance(file_size, (int, float)):
        # If file_size is an integer or float, assume it's in bytes
        return file_size / (1024 ** 2)  # Convert bytes to MB
    else:
        raise TypeError("file_size must be a string or a number")


def is_sequential(numbers: List[int]) -> bool:
    """Check if a list of numbers is sequential."""
    if not numbers:  # Check if the list is empty
        return False

    sorted_numbers = sorted(numbers)  # Sort the numbers
    return all(sorted_numbers[i] + 1 == sorted_numbers[i + 1] 
               for i in range(len(sorted_numbers) - 1))


def get_stat_file_pids(all_files: List[str]) -> List[str]:
    """Extract target tasks from blk_files."""
    target_tasks = set()
    for blk_file in all_files:
        # Get the filename without the path
        filename = os.path.basename(blk_file)
        # replace ".local" for now
        filename = filename.replace(".local", "")
        
        # Split filename by '.'
        parts = filename.split('.')
        if len(parts) >= 3:
            # Expected pattern: <name>.<pid>.<r|w>_blk_trace.json
            # Example: 1-diff....43088-dc065.r_blk_trace.json
            # PID is the third element from the end
            task = parts[-3]
            target_tasks.add(task)
    return sorted(target_tasks)


def add_stat_to_df(trial_folder: str, monitor_timer_stat_io: List, 
                   operation: int, fname: str, task_pid: str, 
                   store_code: int, debug: bool = False) -> Dict[str, Any]:
    """Add statistics to DataFrame from datalife monitor data."""
    # Process the file name
    fname = fname.replace(".local", ".")
    fileName = ".".join(fname.split(".")[:-4])  # Remove last 4 extensions
    fileName = os.path.basename(fileName)      # Keep only the basename
    
    if monitor_timer_stat_io[1] == 0:
        monitor_timer_stat_io[1] = 1  # at least 1 operation if has I/O size
    
    # Initialize statistics
    tmp_write_stat = {
        'aggregateFilesizeMB': bytes_to_mb(monitor_timer_stat_io[2]),
        'transferSize': monitor_timer_stat_io[2] / monitor_timer_stat_io[1],
        'operation': int(operation),
        'totalTime': monitor_timer_stat_io[0],
        'trMiB': bytes_to_mb(monitor_timer_stat_io[2] / monitor_timer_stat_io[0]),
        'storageType': decode_store_code(store_code),  # Convert back to string storage type
        'opCount': monitor_timer_stat_io[1],
        'taskPID': task_pid,
        'fileName': fileName,
    }
    
    if tmp_write_stat['totalTime'] > 100:
        print(f"Recorded large totalTime[{monitor_timer_stat_io}] from task_pid[{task_pid}] fileName[{fileName}]")
    
    # Determine operation type
    op = get_operation_type(operation)
    
    # Ensure the trial folder exists
    if not os.path.exists(trial_folder):
        print(f"Trial folder does not exist: {trial_folder}")
        return tmp_write_stat
    
    # List all files in the trial folder
    all_files = os.listdir(trial_folder)
    
    # Find matching files using substring matching
    matching_files = [
        os.path.join(trial_folder, file)
        for file in all_files
        if f"{fileName}.{task_pid}.local.{op}" in file
    ]
    
    # Process matching files to determine write pattern
    write_pattern = 0  # 0: seq, 1: rand
    for matching_file in matching_files:
        try:
            with open(matching_file) as f:
                w_blk_trace_data = json.load(f)
                blk_list = w_blk_trace_data.get('io_blk_range', [])
                
                # Validate blk_list length
                if len(blk_list) >= 4:
                    if blk_list[3] == -2:
                        write_pattern = 1
                        if debug:
                            print(f"Debug: Detected random write pattern in file: {matching_file}")
                        break
                else:
                    if debug:
                        print(f"Warning: Invalid 'io_blk_range' in file: {matching_file}")
        except Exception as e:
            if debug:
                print(f"Error processing file {matching_file}: {e}")
    
    # Update statistics
    tmp_write_stat['randomOffset'] = write_pattern
    
    return tmp_write_stat


def get_wf_result_df(tests: str, wf_params: List[str], target_tasks: List[str], 
                    storage_type: str = "ssd", debug: bool = False, csv_filename: str = "workflow_data.csv") -> pd.DataFrame:
    """Get workflow result DataFrame from test data."""
    wf_df = pd.DataFrame(columns=wf_params)

    # Identify trial folders
    # FIXME: change this format
    wf_trial_folders = [
        folder for folder in glob.glob(f"{tests}/*")
        if folder.endswith(("t1", "t2", "t3"))
    ]
    
    if debug:
        print(f"Trial folders: {wf_trial_folders}")

    store_code = transform_store_code(storage_type)

    for trial_folder in wf_trial_folders:
        blk_files = glob.glob(f"{trial_folder}/*_blk_trace.json")
        datalife_monitor = glob.glob(f"{trial_folder}/*.datalife.json")
        
        # Check if this is a template workflow with pre-processed CSV data
        workflow_csv = os.path.join(trial_folder, csv_filename)
        if len(blk_files) == 0 and len(datalife_monitor) == 0 and os.path.exists(workflow_csv):
            if debug:
                print(f"WARNING: Loading pre-processed workflow data from: {workflow_csv}")
            try:
                csv_df = pd.read_csv(workflow_csv)
                # Ensure all required columns exist
                for col in wf_params:
                    if col not in csv_df.columns:
                        csv_df[col] = ''
                # Reorder columns to match expected format
                csv_df = csv_df[wf_params]
                # For CSV data, preserve existing task names and stage orders
                if 'taskName' in csv_df.columns and 'stageOrder' in csv_df.columns:
                    # Don't overwrite task names - they're already correct in CSV
                    pass
                wf_df = pd.concat([wf_df, csv_df], ignore_index=True)
                if debug:
                    print(f"Loaded {len(csv_df)} records from CSV")
                continue
            except Exception as e:
                if debug:
                    print(f"Error loading CSV: {e}")
        
        target_tasks = get_stat_file_pids(blk_files)
        if debug:
            print(f"blk_files count: {len(blk_files)}")
            print(f"datalife_monitor count: {len(datalife_monitor)}")
            print(f"target_tasks: {target_tasks}")

        for datalife_json in datalife_monitor:
            task_pid = os.path.basename(datalife_json).split(".")[1]
            if task_pid not in target_tasks:
                continue

            try:
                with open(datalife_json) as f:
                    datalife_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading file: {datalife_json}")
                continue

            task_name = list(datalife_data.keys())[0]
            monitor_timer_stat = datalife_data[task_name]['monitor']
            system_timer_stat = datalife_data[task_name]['system']
            monitor_timer_targets = ["read", "write"]

            for fname in [f for f in blk_files if f".{task_pid}." in f]:
                op_type = "read" if ".r_blk_trace." in fname else "write"
                monitor_stat = monitor_timer_stat[op_type]
                
                # Extra check if monitor_stat has zero value, use value from the other op_type
                for i in range(len(monitor_stat)):
                    if monitor_stat[i] == 0:
                        if op_type == "read":
                            monitor_stat[i] = monitor_timer_stat["write"][i]
                        else:
                            monitor_stat[i] = monitor_timer_stat["read"][i]

                tmp_stat = add_stat_to_df(
                    trial_folder, monitor_stat, 
                    1 if op_type == "read" else 0,
                    fname, task_pid, store_code, debug
                )
                # Use concat instead of _append to avoid fragmentation warnings
                new_row = pd.DataFrame([tmp_stat])
                wf_df = pd.concat([wf_df, new_row], ignore_index=True)

    return wf_df


def get_test_folder_dfs(test_folders: List[str], wf_params: List[str], 
                       target_tasks: List[str], storage_type: str = "pfs", exp_data_path: str = "./", 
                       debug: bool = False, csv_filename: str = "workflow_data.csv") -> pd.DataFrame:
    """Get DataFrame for multiple test folders."""
    folder_dfs = pd.DataFrame(columns=wf_params)
    
    for test_folder in test_folders:
        # Construct the full path for the test folder
        full_test_path = f"{exp_data_path}/{test_folder}"
        wf_df = get_wf_result_df(full_test_path, wf_params, target_tasks, storage_type, debug, csv_filename)
        if len(wf_df) > 0:
            folder_dfs = pd.concat([folder_dfs, wf_df], ignore_index=True)
        
    return folder_dfs


def match_script_name(tests: str) -> Dict[str, Dict[str, Any]]:
    """Match script names and create PID input/output dictionary."""
    # Find folders ending with [t1, t2, t3] in the test_folders
    test_folders = glob.glob(f"{tests}/*")
    wf_trial_folders = [
        folder for folder in test_folders 
        if folder.endswith("t1") or folder.endswith("t2") or folder.endswith("t3")
    ]
    print(f"Trial folders: {wf_trial_folders}")

    pid_input_output_dict = {}

    for trial_folder in wf_trial_folders:
        blk_files = glob.glob(f"{trial_folder}/*_blk_trace.json")
        print(f"len(blk_files) = {len(blk_files)}")
        unique_pids = get_stat_file_pids(blk_files)

        for pid in unique_pids:
            if pid not in pid_input_output_dict:
                pid_input_output_dict[pid] = {
                    "input": [],
                    "output": [],
                    "prevTask": "",
                    "taskName": ""
                }

            # Find the blk_trace_jsons files with the current task_pid
            w_blk_trace_jsons = glob.glob(f"{trial_folder}/*.{pid}.local.w_blk_trace.json")
            r_blk_trace_jsons = glob.glob(f"{trial_folder}/*.{pid}.local.r_blk_trace.json")

            # Replace ".local" with an empty string
            w_blk_trace_jsons = [f.replace(".local", "") for f in w_blk_trace_jsons]
            r_blk_trace_jsons = [f.replace(".local", "") for f in r_blk_trace_jsons]

            # Process write (output) files
            for w_file_path in w_blk_trace_jsons:
                w_file_name_parts = w_file_path.split(".")
                w_file_name = '.'.join(w_file_name_parts[:-3])  # Remove the last 3 extensions
                w_file_basename = os.path.basename(w_file_name)
                pid_input_output_dict[pid]['output'].append(w_file_basename)

            # Process read (input) files
            for r_file_path in r_blk_trace_jsons:
                r_file_name_parts = r_file_path.split(".")
                r_file_name = '.'.join(r_file_name_parts[:-3])  # Remove the last 3 extensions
                r_file_basename = os.path.basename(r_file_name)
                if r_file_basename not in pid_input_output_dict[pid]['input']:
                    pid_input_output_dict[pid]['input'].append(r_file_basename)

    return pid_input_output_dict


def get_wf_pid_script_dict(test_folder: List[str], exp_data_path: str) -> Dict[str, Dict[str, Any]]:
    """Get workflow PID script dictionary for multiple test folders."""
    all_wf_dict = {}

    for tests in test_folder:
        wf_dict = match_script_name(f"{exp_data_path}/{tests}")
        all_wf_dict.update(wf_dict)
    
    return all_wf_dict


def matches_pattern(file_path: str, patterns: List[str]) -> bool:
    """Match a file path against task definition patterns."""
    file_name = os.path.basename(file_path)
    for pattern in patterns:
        try:
            regex_pattern = re.compile(pattern)
            if regex_pattern.fullmatch(file_name):
                return True
        except re.error as e:
            print(f"Invalid regex: {pattern}, Error: {e}")
    return False


def assign_task_names(tasks: Dict[str, Dict[str, Any]], 
                     task_order_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Assign task names and predecessors to tasks based on patterns."""
    for task_pid, details in tasks.items():
        input_paths = details.get('input', [])
        output_paths = details.get('output', [])
        task_name = details.get('taskName', 'unknown')  # Use existing or default to 'unknown'

        # Iterate through each task definition
        for task, definition in task_order_dict.items():
            # Check if any output matches
            if any(matches_pattern(op, definition['outputs']) for op in output_paths):
                task_name = task
                tasks[task_pid]['taskName'] = task_name
                tasks[task_pid]['stage_order'] = definition['stage_order']
                break

            # If no output matches, check for input matches
            for prevTask, predecessor_def in definition['predecessors'].items():
                if any(matches_pattern(ip, predecessor_def.get('inputs', [])) for ip in input_paths):
                    task_name = task
                    tasks[task_pid]['taskName'] = task_name
                    tasks[task_pid]['stage_order'] = definition['stage_order']
                    tasks[task_pid]['prevTask'] = prevTask
                    break

        # If no valid match, warn about the task
        if task_name == 'unknown':
            print(f"Warning: Task PID {task_pid} could not be assigned a valid taskName.")

    return tasks


def _load_workflow_config(wf_name: str) -> Tuple[Dict, Dict, List[int]]:
    """Load workflow configuration and task ordering."""
    config = TEST_CONFIGS[wf_name]
    script_order = config["SCRIPT_ORDER"]
    num_nodes_list = config["NUM_NODES_LIST"]
    exp_data_path = config["exp_data_path"]
    
    # Load task ordering json file
    with open(f"{exp_data_path}/{script_order}.json") as f:
        task_order_dict = json.load(f)
    
    return config, task_order_dict, num_nodes_list


def _assign_basic_task_info(wf_df: pd.DataFrame, all_wf_dict: Dict, task_order_dict: Dict) -> pd.DataFrame:
    """Assign basic task information to DataFrame."""
    # Initialize columns if they don't exist
    if 'prevTask' not in wf_df.columns:
        wf_df['prevTask'] = ""
    if 'taskName' not in wf_df.columns:
        wf_df['taskName'] = "unknown"
    
    # Update DataFrame with task information
    if all_wf_dict:
        for task_pid, task_info in all_wf_dict.items():
            mask = wf_df['taskPID'] == task_pid
            wf_df.loc[mask, 'taskName'] = task_info['taskName']
            wf_df.loc[mask, 'prevTask'] = task_info.get('prevTask', '')
            wf_df.loc[mask, 'stageOrder'] = task_info.get('stage_order', 1)
    
    return wf_df


def _normalize_stage_orders(wf_df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """Normalize stage orders to start from 1."""
    if len(wf_df) > 0:
        min_stage_order = wf_df['stageOrder'].min()
        if min_stage_order == 0:
            if debug:
                print(f"Debug: Found stage 0 in workflow data, normalizing to 1-based (shifting all stages by +1)")
            wf_df['stageOrder'] = wf_df['stageOrder'] + 1
        elif debug:
            print(f"Debug: StageOrder already starts from {min_stage_order}, no normalization needed")
    
    return wf_df


def _assign_prev_tasks_from_patterns(wf_df: pd.DataFrame, task_order_dict: Dict, debug: bool = False) -> pd.DataFrame:
    """Assign prevTask based on file patterns for read operations."""
    if debug:
        print("Debug: Starting prevTask assignment for read operations...")
    
    for index, row in wf_df.iterrows():
        if row['operation'] == 'write':  # Already standardized
            if row['taskName'] == '':
                wf_df.at[index, 'taskName'] = 'none'
        else:  # Read operation
            taskName = row['taskName']
            fileName = row['fileName']
            
            if taskName in task_order_dict:
                task_definition = task_order_dict[taskName]
                if debug:
                    print(f"Debug: Processing {taskName} with fileName {fileName}")
                
                # Check predecessors for this task
                for prevTask, inputs in task_definition['predecessors'].items():
                    input_patterns = inputs['inputs']
                    if debug:
                        print(f"Debug: Checking prevTask {prevTask} with patterns {input_patterns}")
                    if matches_pattern(fileName, input_patterns):
                        if debug:
                            print(f"Debug: MATCH! Setting prevTask to {prevTask} for {fileName}")
                        wf_df.at[index, 'prevTask'] = prevTask
                        break
                    else:
                        if debug:
                            print(f"Debug: No match for {fileName} with patterns {input_patterns}")
            else:
                if debug:
                    print(f"Debug: taskName {taskName} not found in task_order_dict")
    
    return wf_df


def _label_initial_data(wf_df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """Label initial_data for read operations where there's no write operation for the same fileName at stageOrder-1."""
    if len(wf_df) == 0:
        return wf_df
    
    print(f"\n=== _label_initial_data START ===")
    print(f"Input DataFrame shape: {wf_df.shape}")
    
    # Get all read operations with fileName and stageOrder
    read_ops = wf_df[wf_df['operation'] == 'read'][['fileName', 'stageOrder']].copy()
    read_ops['fileName'] = read_ops['fileName'].str.strip()
    
    # Get all write operations with fileName and stageOrder
    write_ops = wf_df[wf_df['operation'] == 'write'][['fileName', 'stageOrder']].copy()
    write_ops['fileName'] = write_ops['fileName'].str.strip()
    
    if debug:
        print(f"Debug: Found {len(read_ops)} read operations and {len(write_ops)} write operations")
        print(f"Debug: Read operations:\n{read_ops}")
        print(f"Debug: Write operations:\n{write_ops}")
    
    # For each read operation, check if there's a write operation for the same fileName at stageOrder-1
    initial_data_mask = wf_df['prevTask'] == ''  # Start with empty prevTask rows
    
    for idx, row in wf_df.iterrows():
        if row['operation'] == 'read' and row['prevTask'] == '':
            file_name = row['fileName'].strip()
            stage_order = row['stageOrder']
            
            # Check if there's a write operation for this fileName at stageOrder-1
            has_write_at_prev_stage = write_ops[
                (write_ops['fileName'] == file_name) & 
                (write_ops['stageOrder'] == stage_order - 1)
            ].shape[0] > 0
            
            if not has_write_at_prev_stage:
                # No write operation at previous stage, this is initial_data
                initial_data_mask.loc[idx] = True
                if debug:
                    print(f"Debug: {file_name} at stage {stage_order} qualifies for initial_data (no write at stage {stage_order-1})")
            else:
                if debug:
                    print(f"Debug: {file_name} at stage {stage_order} does NOT qualify (has write at stage {stage_order-1})")
    
    # Apply the mask to label initial_data
    wf_df.loc[initial_data_mask & (wf_df['operation'] == 'read'), 'prevTask'] = 'initial_data'
    
    # FINAL VERIFICATION: Check what was actually labeled
    labeled_rows = wf_df[wf_df['prevTask'] == 'initial_data']
    print(f"\n=== FINAL CHECK: {len(labeled_rows)} rows labeled as 'initial_data' ===")
    for idx, row in labeled_rows.iterrows():
        print(f"Labeled row {idx}: fileName='{row['fileName']}', operation='{row['operation']}', stageOrder={row['stageOrder']}")
        
        if row['operation'] != 'read':
            print(f"  *** CRITICAL ERROR: Write operation labeled as 'initial_data'! ***")
    
    print(f"=== _label_initial_data END ===\n")
    return wf_df


def _label_final_data(wf_df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """Label final_data for write operations where there's no read operation for the same fileName at stageOrder+1."""
    if len(wf_df) == 0:
        return wf_df
    
    print(f"\n=== _label_final_data START ===")
    print(f"Input DataFrame shape: {wf_df.shape}")
    
    # Get all write operations with fileName and stageOrder
    write_ops = wf_df[wf_df['operation'] == 'write'][['fileName', 'stageOrder']].copy()
    write_ops['fileName'] = write_ops['fileName'].str.strip()
    
    # Get all read operations with fileName and stageOrder
    read_ops = wf_df[wf_df['operation'] == 'read'][['fileName', 'stageOrder']].copy()
    read_ops['fileName'] = read_ops['fileName'].str.strip()
    
    if debug:
        print(f"Debug: Found {len(write_ops)} write operations and {len(read_ops)} read operations")
        print(f"Debug: Write operations:\n{write_ops}")
        print(f"Debug: Read operations:\n{read_ops}")
    
    # For each write operation, check if there's a read operation for the same fileName at stageOrder+1
    final_data_mask = pd.Series([False] * len(wf_df), index=wf_df.index)  # Start with all False
    
    for idx, row in wf_df.iterrows():
        if row['operation'] == 'write':  # Process all write operations regardless of prevTask
            file_name = row['fileName'].strip()
            stage_order = row['stageOrder']
            
            # Check if there's a read operation for this fileName at stageOrder+1
            has_read_at_next_stage = read_ops[
                (read_ops['fileName'] == file_name) & 
                (read_ops['stageOrder'] == stage_order + 1)
            ].shape[0] > 0
            
            if not has_read_at_next_stage:
                # No read operation at next stage, this is final_data
                final_data_mask.loc[idx] = True
                if debug:
                    print(f"Debug: {file_name} at stage {stage_order} qualifies for final_data (no read at stage {stage_order+1})")
            else:
                if debug:
                    print(f"Debug: {file_name} at stage {stage_order} does NOT qualify (has read at stage {stage_order+1})")
    
    # Apply the mask to label final_data (overwrite any existing prevTask values)
    wf_df.loc[final_data_mask & (wf_df['operation'] == 'write'), 'prevTask'] = 'final_data'
    
    # FINAL VERIFICATION: Check what was actually labeled
    labeled_rows = wf_df[wf_df['prevTask'] == 'final_data']
    print(f"\n=== FINAL CHECK: {len(labeled_rows)} rows labeled as 'final_data' ===")
    for idx, row in labeled_rows.iterrows():
        print(f"Labeled row {idx}: fileName='{row['fileName']}', operation='{row['operation']}', stageOrder={row['stageOrder']}")
        
        if row['operation'] != 'write':
            print(f"  *** CRITICAL ERROR: Read operation labeled as 'final_data'! ***")
    
    print(f"=== _label_final_data END ===\n")
    return wf_df


def _add_parallelism_info(wf_df: pd.DataFrame, task_order_dict: Dict, num_nodes_list: List[int]) -> pd.DataFrame:
    """Add parallelism and node information to DataFrame."""
    task_name_to_parallelism = {task: info['parallelism'] for task, info in task_order_dict.items()}
    task_name_to_num_tasks = {task: info['num_tasks'] for task, info in task_order_dict.items()}
    
    for task_name, parallelism in task_name_to_parallelism.items():
        mask = wf_df['taskName'] == task_name
        wf_df.loc[mask, 'parallelism'] = parallelism
        wf_df.loc[mask, 'numTasks'] = task_name_to_num_tasks.get(task_name, 1)
        wf_df.loc[mask, 'numNodesList'] = str(num_nodes_list)
        wf_df.loc[mask, 'numNodes'] = 1
        wf_df.loc[mask, 'tasksPerNode'] = task_name_to_num_tasks.get(task_name, 1)
    
    return wf_df


def _final_verification(wf_df: pd.DataFrame, debug: bool = False) -> None:
    """Perform final verification of labeling."""
    if not debug:
        return
    
    print("\n=== FINAL VERIFICATION ===")
    
    # Check for write operations with initial_data label
    write_with_initial_data = wf_df[
        (wf_df['operation'] == 'write') & 
        (wf_df['prevTask'] == 'initial_data')
    ]
    if len(write_with_initial_data) > 0:
        print(f"ERROR: Found {len(write_with_initial_data)} write operations incorrectly labeled as 'initial_data':")
        for _, row in write_with_initial_data.iterrows():
            print(f"  - {row['fileName']} with operation {row['operation']} at stage {row['stageOrder']}")
    
    # Check for read operations with final_data label
    read_with_final_data = wf_df[
        (wf_df['operation'] == 'read') & 
        (wf_df['prevTask'] == 'final_data')
    ]
    if len(read_with_final_data) > 0:
        print(f"ERROR: Found {len(read_with_final_data)} read operations incorrectly labeled as 'final_data':")
        for _, row in read_with_final_data.iterrows():
            print(f"  - {row['fileName']} with operation {row['operation']} at stage {row['stageOrder']}")
    
    # Summary of labeling
    initial_data_count = len(wf_df[(wf_df['prevTask'] == 'initial_data')])
    final_data_count = len(wf_df[(wf_df['prevTask'] == 'final_data')])
    print(f"Summary: {initial_data_count} rows labeled as 'initial_data', {final_data_count} rows labeled as 'final_data'")


def load_workflow_data(wf_name: str = DEFAULT_WF, debug: bool = False, csv_filename: str = "workflow_data.csv") -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Load and process workflow data.
    
    Parameters:
    - wf_name: Name of the workflow to load
    - debug: Whether to enable debug print statements
    - csv_filename: Name of the CSV file to load (default: "workflow_data.csv")
    
    Returns:
    - DataFrame: Processed workflow data
    - Dict: Task order dictionary
    - Dict: Workflow PID script dictionary
    """
    # Load configuration and task ordering
    config, task_order_dict, num_nodes_list = _load_workflow_config(wf_name)
    
    # Load workflow data
    wf_df = get_test_folder_dfs(
        config["test_folders"], WF_PARAMS, TARGET_TASKS, 
        storage_type="pfs", exp_data_path=config["exp_data_path"], 
        debug=debug, csv_filename=csv_filename
    )
    
    # Get workflow PID script dictionary and assign task names
    all_wf_dict = get_wf_pid_script_dict(config["test_folders"], config["exp_data_path"])
    all_wf_dict = assign_task_names(all_wf_dict, task_order_dict)
    
    # Process workflow data
    wf_df = _assign_basic_task_info(wf_df, all_wf_dict, task_order_dict)
    wf_df = _normalize_stage_orders(wf_df, debug)
    wf_df = _assign_prev_tasks_from_patterns(wf_df, task_order_dict, debug)
    
    # Standardize operations BEFORE labeling
    wf_df['operation'] = wf_df['operation'].apply(standardize_operation)
    if debug:
        print(f"Operations standardized. Operation counts: {wf_df['operation'].value_counts()}")
    
    wf_df = _label_initial_data(wf_df, debug)
    wf_df = _label_final_data(wf_df, debug)
    wf_df = _add_parallelism_info(wf_df, task_order_dict, num_nodes_list)
    
    # Final verification
    _final_verification(wf_df, debug)
    
    # Expand DataFrame for multi-node configurations if enabled
    if MULTI_NODES:
        wf_df = expand_df(wf_df, num_nodes_list)
        if debug:
            print(f"Expanded DataFrame shape after multi-node expansion: {wf_df.shape}")
    
    # Update single-node configurations
    for index, row in wf_df.iterrows():
        if row['parallelism'] == 1:
            wf_df.loc[index, 'numNodes'] = 1
            wf_df.loc[index, 'tasksPerNode'] = 1
    
    return wf_df, task_order_dict, all_wf_dict


def expand_df(wf_df: pd.DataFrame, num_nodes_list: List[int]) -> pd.DataFrame:
    """
    Expand the dataframe for multi-nodes configuration calculation.
    
    This function creates multiple rows for each original row, one for each node count
    in num_nodes_list, and calculates tasksPerNode for each configuration.
    
    Parameters:
    - wf_df: Workflow DataFrame
    - num_nodes_list: List of node counts to expand to
    
    Returns:
    - DataFrame: Expanded DataFrame with rows for each node configuration
    """
    import math

    # If nothing to expand, return early
    if wf_df.empty:
        return wf_df

    # Ensure parallelism column exists and has defaults
    if 'parallelism' not in wf_df.columns:
        wf_df['parallelism'] = 1
    wf_df['parallelism'] = wf_df['parallelism'].fillna(1)

    # Trivial case: only one node (1) and parallelism is 1 across rows ⇒ no expansion
    if len(num_nodes_list) == 1 and num_nodes_list[0] == 1 and (wf_df['parallelism'] == 1).all():
        wf_df['numNodes'] = 1
        wf_df['tasksPerNode'] = 1
        return wf_df.reset_index(drop=True)

    # Create a new DataFrame to store updated rows
    updated_rows = []

    # Iterate through each row in the DataFrame
    for index, row in wf_df.iterrows():
        for num_nodes in num_nodes_list:
            # Create a copy of the current row
            new_row = row.copy()
            try:
                # Update the numNodes and tasksPerNode for the new row
                tasksPerNode = math.ceil(row['parallelism'] / num_nodes)
                new_row['tasksPerNode'] = tasksPerNode
                new_row['numNodes'] = num_nodes
            except Exception as e:
                print(f"Error updating row {index}: {e}: {row}")
                continue
            
            # Append the updated row to the list
            updated_rows.append(new_row)

    # If expansion failed for all rows, apply sensible defaults and return original
    if len(updated_rows) == 0:
        default_nodes = num_nodes_list[0] if len(num_nodes_list) > 0 else 1
        wf_df['numNodes'] = default_nodes
        wf_df['tasksPerNode'] = wf_df['parallelism'].apply(lambda p: math.ceil(p / default_nodes))
        return wf_df.reset_index(drop=True)

    # Create a new DataFrame with the updated rows
    expanded_df = pd.DataFrame(updated_rows)

    # Reset the index of the expanded DataFrame
    expanded_df.reset_index(drop=True, inplace=True)
    
    return expanded_df


def calculate_io_time_breakdown(wf_df: pd.DataFrame, task_name_to_parallelism: Dict[str, int], 
                               num_nodes_list: List[int]) -> Dict[str, float]:
    """Calculate I/O time breakdown per task."""
    # Calculate I/O time per taskName
    write_sub_df = wf_df[wf_df['operation'].apply(lambda x: standardize_operation(x) == 'write')]
    read_sub_df = wf_df[wf_df['operation'].apply(lambda x: standardize_operation(x) == 'read')]

    task_io_time_total = wf_df.groupby('taskName')['totalTime'].sum()
    task_io_time_write = write_sub_df.groupby('taskName')['totalTime'].sum()
    task_io_time_read = read_sub_df.groupby('taskName')['totalTime'].sum()

    task_io_time_adjust = {"read": 0, "write": 0}
    total_wf_io_time = 0
    total_wf_io_time_write = 0
    total_wf_io_time_read = 0
    
    print("Total I/O time per taskName:")
    for task, write_time in task_io_time_write.items():
        # Adjust I/O time by parallelism
        write_time_adjusted = write_time / (task_name_to_parallelism[task] * len(num_nodes_list))
        task_io_time_adjust["write"] += write_time_adjusted
        total_wf_io_time_write += write_time_adjusted
        total_wf_io_time += write_time_adjusted
        print(f" {task} (write): {write_time_adjusted} (sec)")
    
    for task, read_time in task_io_time_read.items():
        # Adjust I/O time by parallelism
        read_time_adjusted = read_time / (task_name_to_parallelism[task] * len(num_nodes_list))
        task_io_time_adjust["read"] += read_time_adjusted
        total_wf_io_time_read += read_time_adjusted
        total_wf_io_time += read_time_adjusted
        print(f" {task} (read): {read_time_adjusted} (sec)")
        
    print(f"Total I/O time per workflow: {total_wf_io_time}")
    
    return {
        'total_io_time': total_wf_io_time,
        'total_write_time': total_wf_io_time_write,
        'total_read_time': total_wf_io_time_read,
        'task_io_time_adjust': task_io_time_adjust
    } 


def standardize_operation(operation):
    """
    Standardize operation codes to string format.
    
    Parameters:
    - operation: Operation code (int or str)
    
    Returns:
    - str: Standardized operation string ('read', 'write', 'cp', 'scp', 'none')
    """
    if isinstance(operation, str):
        # Handle string operations that might be numeric strings
        if operation == '0':
            return 'write'
        elif operation == '1':
            return 'read'
        elif operation == '2':
            return 'cp'
        elif operation == '3':
            return 'scp'
        elif operation == '-1':
            return 'none'
        else:
            return operation
    elif isinstance(operation, (int, np.integer)):
        if operation == 0:
            return 'write'
        elif operation == 1:
            return 'read'
        elif operation == 2:
            return 'cp'
        elif operation == 3:
            return 'scp'
        elif operation == -1:
            return 'none'
        else:
            return str(operation)
    else:
        return str(operation)


def get_operation_type(operation):
    """
    Get operation type for file matching.
    
    Parameters:
    - operation: Operation code (int or str)
    
    Returns:
    - str: Operation type ('w' for write, 'r' for read)
    """
    op = standardize_operation(operation)
    if op in ['write', 'cp']:
        return 'w'
    else:
        return 'r' 