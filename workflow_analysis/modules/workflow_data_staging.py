import pandas as pd
import numpy as np
from collections import defaultdict
from .workflow_data_utils import standardize_operation


def insert_data_staging_rows(wf_df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    Insert data staging (I/O) rows into the workflow DataFrame to simulate data stage_in and stage_out.
    Rules:
    - Initial data movement from beegfs to ssd/tmpfs for stageOrder==0 tasks with operation==1 (read)
    - Intermediate data movement for each unique taskName with stageOrder >=1, for all combinations of [beegfs-ssd, beegfs-tmpfs, ssd-ssd, tmpfs-tmpfs]
    - Final data movement from tmpfs/ssd to beegfs for the last stage
    - Handles splitting by max parallelism of 60 files per row
    - Debug print statements can be toggled with the debug flag
    """
    # Copy to avoid modifying original
    wf_df = wf_df.copy()
    staging_rows = []
    fsblocksize = 4096  # Default block size for transferSize
    max_parallelism = 60

    # Helper: get unique files and their sizes for a set of rows
    def get_file_groups(rows):
        file_groups = []
        files = rows[['taskPID', 'fileName', 'aggregateFilesizeMBtask']].drop_duplicates()
        files = files.reset_index(drop=True)
        for i in range(0, len(files), max_parallelism):
            group = files.iloc[i:i+max_parallelism]
            file_names = group['fileName'].tolist()
            agg_size = group['aggregateFilesizeMBtask'].sum()
            parallelism = len(group)
            file_groups.append((file_names, agg_size, parallelism, group))
        return file_groups

    # 1. Initial data movement (stageOrder==0, operation==1)
    initial_rows = wf_df[(wf_df['stageOrder'] == 0) & (wf_df['operation'].apply(lambda x: standardize_operation(x) == 'read'))]
    if debug:
        print(f"Initial data movement: {len(initial_rows)} rows found.")
    if not initial_rows.empty:
        file_groups = get_file_groups(initial_rows)
        numNodesList = initial_rows['numNodesList'].iloc[0] if 'numNodesList' in initial_rows.columns else [1]
        if isinstance(numNodesList, str):
            try:
                numNodesList = eval(numNodesList)
            except Exception:
                numNodesList = [int(numNodesList)]
        for storageType in ['beegfs-tmpfs', 'beegfs-ssd']:
            for file_names, agg_size, parallelism, group in file_groups:
                for numNodes in numNodesList:
                    row = {
                        'operation': 'cp',
                        'randomOffset': 0,
                        'transferSize': fsblocksize,
                        'aggregateFilesizeMB': agg_size,
                        'numTasks': parallelism,
                        'parallelism': parallelism,
                        'totalTime': '',
                        'numNodesList': numNodesList,
                        'numNodes': numNodes,
                        'tasksPerNode': int(np.ceil(parallelism / numNodes)),
                        'trMiB': '',
                        'storageType': storageType,
                        'opCount': parallelism,
                        'taskName': f'stage_in-0',
                        'taskPID': '',
                        'fileName': ','.join(file_names),
                        'stageOrder': -1,
                        'prevTask': ''
                    }
                    staging_rows.append(row)
                    if debug:
                        print(f"Added initial data movement row: {row}")

    # 2. Intermediate data movement (stageOrder >=1)
    for taskName, group in wf_df[wf_df['stageOrder'] >= 1].groupby('taskName'):
        # Skip if taskName already contains 'stage_out' or 'stage_in'
        if 'stage_out' in taskName or 'stage_in' in taskName:
            continue
        stageOrder = group['stageOrder'].iloc[0]
        prevTask = group['prevTask'].iloc[0] if 'prevTask' in group.columns else ''
        file_groups = get_file_groups(group)
        numNodesList = group['numNodesList'].iloc[0] if 'numNodesList' in group.columns else [1]
        if isinstance(numNodesList, str):
            try:
                numNodesList = eval(numNodesList)
            except Exception:
                numNodesList = [int(numNodesList)]
        for storageType in ['beegfs-ssd', 'beegfs-tmpfs', 'ssd-ssd', 'tmpfs-tmpfs']:
            # stage_in
            for file_names, agg_size, parallelism, file_group in file_groups:
                for numNodes in numNodesList:
                    op = 'scp' if storageType in ['ssd-ssd', 'tmpfs-tmpfs'] else 'cp'
                    row = {
                        'operation': op,
                        'randomOffset': 0,
                        'transferSize': fsblocksize,
                        'aggregateFilesizeMB': agg_size,
                        'numTasks': parallelism,
                        'parallelism': parallelism,
                        'totalTime': '',
                        'numNodesList': numNodesList,
                        'numNodes': numNodes,
                        'tasksPerNode': int(np.ceil(parallelism / numNodes)),
                        'trMiB': '',
                        'storageType': storageType,
                        'opCount': parallelism,
                        'taskName': f'stage_in-{taskName}',
                        'taskPID': '',
                        'fileName': ','.join(file_names),
                        'stageOrder': stageOrder - 0.5,
                        'prevTask': prevTask
                    }
                    staging_rows.append(row)
                    if debug:
                        print(f"Added intermediate stage_in row: {row}")
            # stage_out
            for file_names, agg_size, parallelism, file_group in file_groups:
                for numNodes in numNodesList:
                    op = 'scp' if storageType in ['ssd-ssd', 'tmpfs-tmpfs'] else 'cp'
                    row = {
                        'operation': op,
                        'randomOffset': 0,
                        'transferSize': fsblocksize,
                        'aggregateFilesizeMB': agg_size,
                        'numTasks': parallelism,
                        'parallelism': parallelism,
                        'totalTime': '',
                        'numNodesList': numNodesList,
                        'numNodes': numNodes,
                        'tasksPerNode': int(np.ceil(parallelism / numNodes)),
                        'trMiB': '',
                        'storageType': storageType,
                        'opCount': parallelism,
                        'taskName': f'stage_out-{taskName}',
                        'taskPID': '',
                        'fileName': ','.join(file_names),
                        'stageOrder': stageOrder + 0.5,
                        'prevTask': taskName
                    }
                    staging_rows.append(row)
                    if debug:
                        print(f"Added intermediate stage_out row: {row}")

    # 2b. Insert stage-out for all tasks with write operations (operation == 0)
    write_rows = wf_df[wf_df['operation'].apply(lambda x: standardize_operation(x) == 'write')]
    for taskName, group in write_rows.groupby('taskName'):
        # Skip if taskName already contains 'stage_out' or 'stage_in'
        if 'stage_out' in taskName or 'stage_in' in taskName:
            continue
        stageOrder = group['stageOrder'].iloc[0]
        file_groups = get_file_groups(group)
        numNodesList = group['numNodesList'].iloc[0] if 'numNodesList' in group.columns else [1]
        if isinstance(numNodesList, str):
            try:
                numNodesList = eval(numNodesList)
            except Exception:
                numNodesList = [int(numNodesList)]
        for storageType in ['beegfs-ssd', 'beegfs-tmpfs', 'ssd-ssd', 'tmpfs-tmpfs']:
            for file_names, agg_size, parallelism, file_group in file_groups:
                for numNodes in numNodesList:
                    op = 'scp' if storageType in ['ssd-ssd', 'tmpfs-tmpfs'] else 'cp'
                    row = {
                        'operation': op,
                        'randomOffset': 0,
                        'transferSize': fsblocksize,
                        'aggregateFilesizeMB': agg_size,
                        'numTasks': parallelism,
                        'parallelism': parallelism,
                        'totalTime': '',
                        'numNodesList': numNodesList,
                        'numNodes': numNodes,
                        'tasksPerNode': int(np.ceil(parallelism / numNodes)),
                        'trMiB': '',
                        'storageType': storageType,
                        'opCount': parallelism,
                        'taskName': f'stage_out-{taskName}',
                        'taskPID': '',
                        'fileName': ','.join(file_names),
                        'stageOrder': stageOrder + 0.5,
                        'prevTask': taskName
                    }
                    staging_rows.append(row)
                    if debug:
                        print(f"Added stage_out row for write op: {row}")

    # 3. Final data movement (last stage)
    max_stage = wf_df['stageOrder'].max()
    last_rows = wf_df[wf_df['stageOrder'] == max_stage]
    if debug:
        print(f"Final data movement: {len(last_rows)} rows found for stageOrder {max_stage}.")
    if not last_rows.empty:
        file_groups = get_file_groups(last_rows)
        numNodesList = last_rows['numNodesList'].iloc[0] if 'numNodesList' in last_rows.columns else [1]
        if isinstance(numNodesList, str):
            try:
                numNodesList = eval(numNodesList)
            except Exception:
                numNodesList = [int(numNodesList)]
        for storageType in ['tmpfs-beegfs', 'ssd-beegfs']:
            for file_names, agg_size, parallelism, group in file_groups:
                for numNodes in numNodesList:
                    # Skip if taskName already contains 'stage_out' or 'stage_in'
                    for taskName in last_rows['taskName'].unique():
                        if 'stage_out' in taskName or 'stage_in' in taskName:
                            continue
                        row = {
                            'operation': 'cp',
                            'randomOffset': 0,
                            'transferSize': fsblocksize,
                            'aggregateFilesizeMB': agg_size,
                            'numTasks': parallelism,
                            'parallelism': parallelism,
                            'totalTime': '',
                            'numNodesList': numNodesList,
                            'numNodes': numNodes,
                            'tasksPerNode': int(np.ceil(parallelism / numNodes)),
                            'trMiB': '',
                            'storageType': storageType,
                            'opCount': parallelism,
                            'taskName': f'stage_out-{taskName}',
                            'taskPID': '',
                            'fileName': ','.join(file_names),
                            'stageOrder': max_stage + 0.5,
                            'prevTask': taskName,
                        }
                        staging_rows.append(row)
                        if debug:
                            print(f"Added final data movement row: {row}")

    # Combine and sort
    staging_df = pd.DataFrame(staging_rows)
    combined = pd.concat([wf_df, staging_df], ignore_index=True, sort=False)
    combined = combined.sort_values(['stageOrder', 'taskName']).reset_index(drop=True)
    if debug:
        print(f"Total rows after staging: {len(combined)}")
    return combined 