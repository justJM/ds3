#%%
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_nfl_table(url,limit=None):
    response = requests.get(url)
    soup = BeautifulSoup(response.text,'html.parser')
    table = soup.select_one('#data')
    headers = [header.text for header in table.find_all('th')]    
    body_trs = table.find('tbody').find_all('tr',limit=limit)
    records = [{key:to_numeric(rec.string) for key,rec in zip(headers,record.find_all('td'))} for record in body_trs]
    return records

def to_numeric(s):
    try:
        conv = float(s)
        i = int(conv)
        if i== conv:
            return i
        return conv
    except ValueError:
        return s


url = 'https://www.fantasypros.com/nfl/reports/leaders/rb.php?year=2015'
table = pd.DataFrame(get_nfl_table(url,limit=10))
player = table['Player']
avg= table['Avg']
#%%
#tensorflow