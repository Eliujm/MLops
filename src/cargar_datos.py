def cargar_datos(path_file):
    

    import pandas as pd 

    try: 
        if path_file.endswith('.xlsx'):
            df = pd.read_excel(path_file)
        else:
            df = pd.read_csv(path_file)
        return df
    except FileNotFoundError:
        print(f'please verify the path file, {path_file} does not exist')
        
    