"""Run inside an API container: read no rows, print only table/status."""
import os
import requests

url = os.environ['SUPABASE_URL'].rstrip('/')
key = os.environ.get('SUPABASE_SERVICE_KEY') or os.environ['SUPABASE_SERVICE_ROLE_KEY']
for table in ('dataset_builds', 'model_versions', 'predictions', 'training_queue', 'training_state'):
    response = requests.get(url + '/rest/v1/' + table,
                            headers={'apikey': key, 'Authorization': 'Bearer ' + key},
                            params={'select': '*', 'limit': 0}, timeout=30)
    print(table, response.status_code)
    if response.status_code != 200 or response.json() != []:
        raise SystemExit('Read-only service-role check failed')
