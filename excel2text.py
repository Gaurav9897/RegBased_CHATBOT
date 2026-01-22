import pandas as pd
def row_extractor(data):
    context = ''
    for idx,row in data.iterrows():
        row_text = '|'.join(f"Index:{idx} {col}:{row[col]}" for col in data.columns if pd.notna(row[col]))
        context =  context + row_text + '\n'
        return context