import json
import logging
from pathlib import Path
from typing import List, Dict

import requests
import pandas as pd
from requests.auth import HTTPBasicAuth
from utils.config_loader import ConfigLoader

# --- 目录 ---
BASE_DIR = Path(__file__).resolve().parents[1]
WQ_FIELD_DIR = BASE_DIR / "data" / "wq_fields"
WQ_FIELD_DIR.mkdir(parents=True, exist_ok=True)
WQ_OPERATOR_DIR = BASE_DIR / "data" / "wq_operators"
WQ_OPERATOR_DIR.mkdir(parents=True, exist_ok=True)

FIELDS_CSV = WQ_FIELD_DIR
OPERATORS_CSV = WQ_OPERATOR_DIR / "operators.csv"


class OpAndFeature:
    def __init__(self):
        self.sess = requests.Session()
        username = ConfigLoader.get("worldquant_account")
        password = ConfigLoader.get("worldquant_password")
        self.setup_auth(username, password)

    def setup_auth(self, username, password) -> None:
        """Set up authentication with WorldQuant Brain."""
        self.sess.auth = HTTPBasicAuth(username, password)

        print("Authenticating with WorldQuant Brain...")
        response = self.sess.post('https://api.worldquantbrain.com/authentication')
        print(f"Authentication response status: {response.status_code}")
        logging.debug(f"Authentication response: {response.text[:500]}...")

        if response.status_code != 201:
            raise Exception(f"Authentication failed: {response.text}")

    def get_data_fields(self):
        """Fetch available data fields from WorldQuant Brain across multiple datasets with random sampling."""

        # datasets = ['pv1', 'fundamental6', 'analyst4', 'model16', 'news12']

        datasets = ['analyst4',
                    'analyst10',
                    'analyst11',
                    'analyst14',
                    'analyst15',
                    'analyst16',
                    'analyst35',
                    'analyst40',
                    'analyst69',
                    'earnings3',
                    'earnings5',
                    'fundamental17',
                    'fundamental22',
                    'fundamental23',
                    'fundamental28',
                    'fundamental31',
                    'fundamental44',
                    'fundamental6',
                    'fundamental7',
                    'fundamental72',
                    'model109',
                    'model110',
                    'model138',
                    'model16',
                    'model176',
                    'model219',
                    'model238',
                    'model244',
                    'model26',
                    'model262',
                    'model264',
                    'model29',
                    'model30',
                    'model307',
                    'model32',
                    'model38',
                    'model53',
                    'model77',
                    'news12',
                    'news20',
                    'news23',
                    'news52',
                    'news66',
                    'other128',
                    'other432',
                    'other450',
                    'other455',
                    'other460',
                    'other496',
                    'other551',
                    'other553',
                    'other699',
                    'other83',
                    'pv1',
                    'pv13',
                    'pv173',
                    'pv29',
                    'pv37',
                    'pv53',
                    'pv72',
                    'pv73',
                    'pv96',
                    'risk60',
                    'risk66',
                    'risk70',
                    'sentiment21',
                    'sentiment22',
                    'sentiment26',
                    'shortinterest6',
                    'univ1']

        base_params = {
            'delay': 1,
            'instrumentType': 'EQUITY',
            'limit': 50,
            'region': 'USA',
            'universe': 'TOP3000'
        }

        try:
            print("Requesting data fields from multiple datasets...")
            for dataset in datasets:
                print("------------" + dataset + "--------------\n")
                des = str(FIELDS_CSV) + "/" + dataset + ".csv"
                if Path(des).exists():
                    print(f"Fields CSV already exists at {des}, skipping download.")
                    continue

                all_fields = []
                params = base_params.copy()
                params['dataset.id'] = dataset

                print(f"Getting field count for dataset: {dataset}")
                count_response = self.sess.get('https://api.worldquantbrain.com/data-fields', params=params)

                if count_response.status_code == 200:
                    count_data = count_response.json()
                    total_fields = count_data.get('count', 0)
                    print(f"Total fields in {dataset}: {total_fields}")

                    params['limit'] = 50

                    for offset in range(0, total_fields, params['limit']):
                        params['offset'] = offset
                        response = self.sess.get('https://api.worldquantbrain.com/data-fields', params=params)

                        if response.status_code == 200:
                            data = response.json()
                            fields = data.get('results', [])
                            print(f"Fetched {len(fields)} fields at offset={offset}")
                            all_fields.extend(fields)
                        else:
                            print(f"Failed to fetch fields for {dataset} at offset={offset}: {response.text[:500]}")
                else:
                    print(f"Failed to get count for {dataset}: {count_response.text[:500]}")

                # 去重
                unique_fields = {field['id']: field for field in all_fields}.values()
                unique_fields = list(unique_fields)

                # 保存到 CSV
                df = pd.DataFrame(unique_fields)
                df.to_csv(des, index=False, encoding='utf-8')
                print(f"✅ Saved fields CSV to {des}")

            return

        except Exception as e:
            print(f"Failed to fetch data fields: {e}")
            return

    def get_operators(self) -> List[Dict]:
        """Fetch available operators from WorldQuant Brain."""
        if OPERATORS_CSV.exists():
            print(f"Operators CSV already exists at {OPERATORS_CSV}, skipping download.")
            return pd.read_csv(OPERATORS_CSV).to_dict(orient='records')

        print("Requesting operators...")
        response = self.sess.get('https://api.worldquantbrain.com/operators')
        print(f"Operators response status: {response.status_code}")
        logging.debug(f"Operators response: {response.text[:500]}...")  # Print first 500 chars

        if response.status_code != 200:
            raise Exception(f"Failed to get operators: {response.text}")

        data = response.json()
        if isinstance(data, list):
            operators = data
        elif 'results' in data:
            operators = data['results']
        else:
            raise Exception(f"Unexpected operators response format. Response: {data}")

        df = pd.DataFrame(operators)
        df.to_csv(OPERATORS_CSV, index=False, encoding='utf-8')
        print(f"✅ Saved operators CSV to {OPERATORS_CSV}")

        return operators
