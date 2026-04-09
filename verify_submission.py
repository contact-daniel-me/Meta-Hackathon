import json
import os

def is_score_like(key, val):
    # Exclude certain fields that are allowed to be outside (0,1)
    exclude_keys = ['id', 'seed', 'latitude', 'longitude', 'lat', 'lon', 'step', 'steps_taken', 'max_steps', 'budget', 'time_remaining', 'battery_capacity_kwh', 'power_kw', 'waiting_time_minutes', 'queue_length', 'step_count']
    if any(ex in key.lower() for ex in exclude_keys):
        return False
    return isinstance(val, (float, int)) and not isinstance(val, bool)

def check_recursive(data, path=""):
    all_good = True
    if isinstance(data, dict):
        for k, v in data.items():
            new_path = f"{path}.{k}" if path else k
            if is_score_like(k, v):
                if not (0.001 <= v <= 0.999):
                    print(f"  FAIL: {new_path} = {v} (out of range)")
                    all_good = False
            elif not check_recursive(v, new_path):
                all_good = False
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if not check_recursive(item, f"{path}[{i}]"):
                all_good = False
    return all_good

def check_submission():
    if not os.path.exists('submission.json'):
        print('Error: submission.json not found')
        return False
        
    try:
        with open('submission.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f'Error loading JSON: {e}')
        return False
        
    print(f'Performing deep recursive validation...')
    return check_recursive(data)

if __name__ == '__main__':
    if check_submission():
        print('\nOVERALL RESULT: SUCCESS - All score-like values are strictly compliant (0.001, 0.999)')
    else:
        print('\nOVERALL RESULT: FAILURE - Found values outside range')
        exit(1)

if __name__ == '__main__':
    if check_submission():
        print('\nOVERALL RESULT: SUCCESS - All scores are in strictly compliant range (0.001, 0.999)')
    else:
        print('\nOVERALL RESULT: FAILURE - One or more values fall outside range')
        exit(1)
