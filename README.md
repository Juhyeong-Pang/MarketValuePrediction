# Project Overview

Last Modified: 03/06/2026

### Motivation

We wanted to practice data collection, preprocessing, and analysis. As footballs (soccer) are an economically significant industry, it generates vast amounts of numerical data. Therefore, we thought leveraging football data will provide an experience where we can leverage skills required to handle complex, real-world datasets.

### Objective

To create a model that predicts a player’s market value based on their performance stats and generate insights on which features have the greatest influence.

# The Approach and Process

## 1. Collect Data

- Performance Stat: Used Selenium and BeautifulSoup for collecting data from [fbref.com](http://fbref.com).
    - Created an algorithm that finds links that contain table with performance stats for players in the Big 5 League in Europe
    - Entered destination link using Selenium and extracted HTML.
    - Used BeautifulSoup to extract table from the HTML.
- Market Value: Used Request and BeautifulSoup for collecting data from [transfermarkt.com](http://transfermarkt.com)
    - Created an algorithm that finds links that contain market values for players in the Big 5 League in Europe
    - Used Request and BeautifulSoup to extract data from the sites.
- Concatenated Both Dataset based on player’s name
- View Code
    
    
    ```python
    # Collect Performance Stats
    def click_link(driver, link, wait_sec=3):
        driver.execute_script("arguments[0].scrollIntoView();", link)
        time.sleep(wait_sec)
        driver.execute_script("arguments[0].click();", link)
        print(f"Current URL: {driver.current_url}")
    
    def stat_get_html(driver, wait, league, season):
        index = 2026 - season
        clicks = 0
        
        link = wait.until(EC.presence_of_element_located((By.LINK_TEXT, league)))
        click_link(driver, link, wait_sec=7)
        clicks += 1
        
        link = wait.until(EC.presence_of_element_located((By.XPATH, f'//*[@id="seasons"]/tbody/tr[{index}]/td[1]/a')))
        click_link(driver, link, wait_sec=3)
        clicks += 1
    
        link = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="inner_nav"]/ul/li[2]/div/ul[1]/li[1]/a')))
        click_link(driver, link, wait_sec=2)
        clicks += 1
    
        time.sleep(5)
        print(f"Successfully arrived! Current URL: {driver.current_url}")
        html = driver.page_source
    
        for _ in range(clicks):
            driver.back()
            print(f"Returned to: {driver.current_url}")
            time.sleep(1)
    
        return html
    
    # season: 2025->1, 2024->2, ...
    def get_stats_df(season):
        league_list = ["Premier League", "La Liga", "Ligue 1", "Fußball-Bundesliga", "Serie A"]
        final_df = None
        html_list = []
        
        driver = uc.Chrome(version_main=144)
    
        try:
            for league in league_list:
                driver.get("https://fbref.com/")
    
                wait = WebDriverWait(driver, 15)
                
                comps_link = wait.until(EC.presence_of_element_located((By.LINK_TEXT, "Competitions")))
                click_link(driver, comps_link, wait_sec=2)
                
                html = stat_get_html(driver, wait, league, season)
                html_list.append(html)
                print(f"Added html to the list, currently have {len(html_list)} htmls")
    
        except Exception as e:
            print(f"An error occurred: {e}")
        
        finally:
            time.sleep(5)
            driver.quit()
    
        if html_list:
            df_list = []
            for html in html_list:
                soup = BeautifulSoup(html, 'html')
                table = soup.find('table', id='stats_standard')
                
                if table:
                    df = pd.read_html(StringIO(str(table)))[0]
                    
                    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]
                    df_list.append(df)
                else:
                    print("Could not find the stats table.")
            final_df = pd.concat(df_list, ignore_index=True).rename(columns={'Unnamed: 1_level_0_Player': 'Name'})
            final_df.to_csv(f"stats{season}.csv")
        else:
            print("html not found")
    
        return final_df
    ```
    
    ```python
    # Collect Market Value
    def get_response(url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
    
        return response
    
    def create_url(team_name, team_id, season):
        return f"https://www.transfermarkt.us/{team_name}/startseite/verein/{team_id}/saison_id/{season}"
    
    def parse_value(text):
        num_part = re.search(r'[0-9.]+', text).group()
        value = float(num_part)
    
        if 'm' in text:
            return value * 1_000_000
        elif 'k' in text:
            return value * 1_000
        elif 'bn' in text:
            return value * 1_000_000_000
        return value
    
    def get_values_df(season):
        league_url_list = [
            "https://www.transfermarkt.us/premier-league/startseite/wettbewerb/GB1", 
            "https://www.transfermarkt.us/laliga/startseite/wettbewerb/ES1", 
            "https://www.transfermarkt.us/serie-a/startseite/wettbewerb/IT1", 
            "https://www.transfermarkt.us/bundesliga/startseite/wettbewerb/L1", 
            "https://www.transfermarkt.us/ligue-1/startseite/wettbewerb/FR1", 
        ]
        url_list = []
        
        team_url_list = []
        for league_url in league_url_list:
            print(f"going into: {league_url}")
            response = get_response(league_url)
            time.sleep(3)
        
            if response.status_code == 200:
                tree = html.fromstring(response.content) 
                rows = tree.xpath('//*[@id="yw1"]/table/tbody/tr')
                for row in rows:
                    links = row.xpath('./td[2]/a/@href')
                    if links:
                        full_url = "https://www.transfermarkt.us" + links[0] + ""
                        team_url_list.append(full_url)
    
        print(f"Found {len(team_url_list)} Teams.")
        
        team_name_id_map = {}
        
        for team_url in team_url_list:
            team_url_elements = team_url.split("/")
            team_name = team_url_elements[team_url_elements.index("www.transfermarkt.us") + 1]
            team_id = team_url_elements[team_url_elements.index("verein") + 1]
            team_name_id_map[team_name] = team_id
    
        for team_name in team_name_id_map:
            url_list.append(create_url(team_name, team_name_id_map[team_name], season))
    
        name_to_value = {}
    
        num = 0
        length = len(url_list)
        for url in url_list:
            # print(f"going into: {url}")
            response = get_response(url)
            s = 3
            time.sleep(s)
        
            if response.status_code == 200:
                tree = html.fromstring(response.content)
                rows = tree.xpath('//*[@id="yw1"]/table/tbody/tr')
                for row in rows: 
                    name_list = row.xpath('.//td[2]//table//tr[1]//td[2]//a/text()')
                    value_list = row.xpath('./td[5]/a/text()')
                    if name_list and value_list:
                        name_text = name_list[0].strip()
                        value = parse_value(value_list[0].strip())
                        name_to_value[name_text] = value
            num += 1
            print(f"{num}/{length} Teams Added, Estimated Time Remaining: {(length - num) * (s + 1)}s")
        final_df = pd.DataFrame(list(name_to_value.items()), columns=['Name', 'Value (€)'])
        final_df.to_csv(f"values{season}.csv")
        return final_df
    ```
    
    ```python
    # Concatenate Two Datasets
    def normalize_name(name):
        if not isinstance(name, str): return ""
        name = unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode("utf-8")
        return name.lower().strip()
    
    def normalize_and_merge(stats_df, name_value_df):
        stats_df['Name_Clean'] = stats_df['Name'].apply(normalize_name)
        name_value_df['Name_Clean'] = name_value_df['Name'].apply(normalize_name)
    
        return pd.merge(stats_df.reset_index(drop=True), 
                        name_value_df.reset_index(drop=True), 
                        on='Name_Clean', 
                        how='left')
    
    def get_dataset(season):
        stats_df = get_stats_df(season)
        value_df = get_values_df(season)
        merged_df = normalize_and_merge(stats_df=stats_df, name_value_df=value_df)
        merged_df.to_csv(f"{season}.csv")
    
        return merged_df
    ```
    

## 2. Process Data

- Renamed column names and dropped meaningless columns + Changed string type into numeric type
    - View Code
        
        ```python
        def clean_dataframe(df):
            df = df.dropna(subset = ["Value (€)"])
            df = df.drop(columns = ["Name_Clean", 
                                    "Unnamed: 0_y", 
                                    "Name_y", 
                                    "Unnamed: 0_x", 
                                    "Unnamed: 0_level_0_Rk", 
                                    "Unnamed: 24_level_0_Matches",
                                    "Unnamed: 0"
                                   ])
            df = df.rename(columns={
                'Unnamed: 3_level_0_Pos': 'Position', 
                'Unnamed: 5_level_0_Age': 'Age',
                'Unnamed: 6_level_0_Born': 'Born',
                'Unnamed: 2_level_0_Nation': 'Nationality',
                'Playing Time_MP': 'Match Played',
                'Name_x': 'Name',
                'Playing Time_Starts': 'Match Started',
                'Playing Time_Min': 'Minutes Played',
                'Playing Time_90s': 'Minutes Played / 90',
                'Performance_Gls': 'Goals',
                'Performance_Ast': 'Assists',
                'Performance_G+A': 'Goals + Assists',
                'Performance_G-PK': 'Non-Penality Goals',
                'Performance_PK': 'Penalty Kick Goals',
                'Performance_PKatt': 'Penalty Kick Attempted',
                'Performance_CrdY': 'Yellow Cards',
                'Performance_CrdR': 'Red Cards',
                'Per 90 Minutes_Gls': 'Goals Per 90 Minutes',
                'Per 90 Minutes_Ast': 'Assists Per 90 Minutes',
                'Per 90 Minutes_G+A': 'G+A Per 90 Minutes',
                'Per 90 Minutes_G-PK': 'Non-Penality Goals Per 90 Minutes',
                'Per 90 Minutes_G+A-PK': 'Non-Penalty Goals + Assists/90',
                'Value (€)': 'Value',
                # '': '',
                # '': '',
                # '': '',
                # '': '',
                # '': '',
                # '': '',
            })
        
            intger_column_list = ["Age", "Born", "Match Played", "Match Started", "Minutes Played", "Goals", "Assists", "Goals + Assists", 
                          "Non-Penality Goals", "Penalty Kick Goals", "Penalty Kick Attempted", "Yellow Cards", "Red Cards"]
            
            float_column_list = ["Minutes Played / 90", "Goals Per 90 Minutes", "Assists Per 90 Minutes", 
                          "G+A Per 90 Minutes", "Non-Penality Goals Per 90 Minutes", "Non-Penalty Goals + Assists/90"]
        
            numeric_column_list = []
            numeric_column_list.extend(intger_column_list)
            numeric_column_list.extend(float_column_list)
        
            for column in numeric_column_list:
                df[column] = pd.to_numeric(df[column], errors='coerce')
            
            return df
        ```
        
- Performed one-hot encoding for Position, and scaled the data using StandardScaler() + Dropped columns for Nationality and Year Born.
    - Scaled the data since most columns were skewed to the left → Finding a way to address data imbalance would be better than scaling (ex. Oversampling)
    - View Code
        
        ```python
        def preprocess_dataframe(df):
            df = df.copy()
        
            df_rest = df.drop(columns=["Value"])
            
            scaler = StandardScaler()
            numeric_cols = df_rest.select_dtypes(include=['float64', 'int64']).columns
            scaler.fit(df[numeric_cols])
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
            position_dummies = df['Position'].str.get_dummies(sep=',')
            df = pd.concat([df, position_dummies], axis=1)   
            df = df.set_index('Name')
            df = df.drop(columns = ["Nationality", "Born", "Position"])
        
            df = df[[c for c in df.columns if c != "Value"] + ["Value"]]
            return df, scaler
        ```
        
- Changed the Squad into the squad’s final rank after the season.
    - View Code
        
        ```python
        squad_rank_map = {
            # [21-22, 22-23, 23-24, 24-25]
            'Ajaccio': [22, 18, 22, 22], 'Alavés': [20, 22, 10, 12], 'Almería': [22, 17, 19, 22], 'Angers': [14, 20, 22, 18],
            'Arminia': [17, 22, 22, 22], 'Arsenal': [5, 2, 2, 2], 'Aston Villa': [14, 7, 4, 5], 'Atalanta': [8, 5, 4, 3],
            'Athletic Club': [8, 8, 5, 6], 'Atlético Madrid': [3, 3, 4, 3], 'Augsburg': [14, 15, 11, 13], 'Auxerre': [22, 17, 22, 15],
            'Barcelona': [2, 1, 2, 1], 'Bayern Munich': [1, 1, 3, 1], 'Bochum': [13, 14, 16, 17], 'Bologna': [13, 9, 5, 8],
            'Bordeaux': [20, 22, 22, 22],'Bournemouth': [22, 15, 12, 11],'Brentford': [13, 9, 16, 14],'Brest': [11, 14, 3, 9],
            'Brighton': [9, 6, 11, 10],'Burnley': [18, 22, 19, 22],'Cagliari': [18, 22, 16, 16],'Celta Vigo': [11, 13, 14, 11],
            'Chelsea': [3, 12, 6, 4],'Clermont Foot': [17, 8, 18, 22],'Cremonese': [22, 19, 22, 22],'Crystal Palace': [12, 11, 10, 12],
            'Cádiz': [17, 14, 18, 22],'Darmstadt 98': [22, 22, 18, 22],'Dortmund': [2, 2, 5, 4],'Eintracht Frankfurt': [11, 7, 6, 6],
            'Elche': [13, 20, 22, 22],'Empoli': [14, 14, 17, 14],'Espanyol': [14, 19, 22, 17],'Everton': [16, 17, 15, 15],
            'Fiorentina': [7, 8, 8, 7], 'Freiburg': [6, 5, 10, 8], 'Frosinone': [22, 22, 18, 22],
            'Fulham': [22, 10, 13, 12], 'Genoa': [19, 22, 11, 13], 'Getafe': [15, 15, 12, 14], 'Girona': [22, 10, 3, 7],
            'Gladbach': [10, 10, 14, 11], 'Granada': [18, 22, 20, 22], 'Greuther Fürth': [18, 22, 22, 22], 'Heidenheim': [22, 22, 8, 12],
            'Hellas Verona': [9, 18, 13, 15], 'Hertha BSC': [16, 18, 22, 22], 'Hoffenheim': [9, 12, 7, 10], 'Inter': [2, 3, 1, 2],
            'Juventus': [4, 7, 3, 3], 'Köln': [7, 11, 17, 22], 'Las Palmas': [22, 22, 16, 17], 'Lazio': [5, 2, 7, 6],
            'Le Havre': [22, 22, 15, 16], 'Lecce': [22, 16, 14, 15], 'Leeds United': [17, 19, 22, 22], 'Leicester City': [8, 18, 22, 16],
            'Lens': [7, 2, 7, 8], 'Levante': [19, 22, 22, 22], 'Leverkusen': [3, 6, 1, 3], 'Lille': [10, 5, 4, 5],
            'Liverpool': [2, 5, 3, 1], 'Lorient': [16, 10, 19, 22], 'Luton Town': [22, 22, 18, 22], 'Lyon': [8, 7, 6, 6],
            'Mainz 05': [8, 9, 13, 12], 'Mallorca': [16, 9, 15, 13], 'Manchester City': [1, 1, 1, 2], 'Manchester Utd': [6, 3, 8, 7],
            'Marseille': [2, 3, 8, 5], 'Metz': [19, 22, 18, 22], 'Milan': [1, 4, 2, 4], 'Monaco': [3, 6, 2, 2],
            'Montpellier': [13, 12, 12, 14], 'Monza': [22, 11, 12, 15], 'Nantes': [9, 16, 14, 16], 'Napoli': [3, 1, 10, 4],
            'Newcastle United': [11, 4, 7, 6], 'Nice': [5, 9, 5, 6], 'Norwich City': [20, 22, 22, 22], 'Nottingham Forest': [22, 16, 17, 14],
            'Osasuna': [10, 7, 11, 10], 'Paris Saint-Germain': [1, 1, 1, 1], 'RB Leipzig': [4, 3, 4, 4], 'Rayo Vallecano': [12, 11, 17, 15],
            'Real Betis': [5, 6, 7, 7], 'Real Madrid': [1, 2, 1, 2], 'Real Sociedad': [6, 4, 6, 8], 'Reims': [12, 11, 9, 10],
            'Rennes': [4, 4, 10, 9], 'Roma': [6, 6, 6, 7], 'Saint-Étienne': [18, 22, 22, 16], 'Salernitana': [17, 15, 20, 22],
            'Sampdoria': [15, 20, 22, 22], 'Sassuolo': [11, 13, 19, 22], 'Schalke 04': [22, 17, 22, 22], 'Sevilla': [4, 12, 14, 12],
            'Sheffield United': [22, 22, 20, 22], 'Southampton': [15, 20, 22, 19], 'Spezia': [16, 18, 22, 22],'Strasbourg': [6, 15, 13, 13],
            'Stuttgart': [15, 16, 2, 7],'Torino': [10, 10, 9, 11], 'Tottenham Hotspur': [4, 8, 5, 5],'Toulouse': [22, 13, 11, 12],
            'Troyes': [15, 19, 22, 22],'Udinese': [12, 12, 15, 13], 'Union Berlin': [5, 4, 15, 11],'Valencia': [9, 14, 9, 13],
            'Valladolid': [22, 18, 22, 19],'Venezia': [20, 22, 22, 20], 'Villarreal': [7, 5, 8, 6],'Watford': [19, 22, 22, 22],
            'Werder Bremen': [22, 13, 9, 10],'West Ham United': [7, 14, 9, 11], 'Wolfsburg': [12, 8, 12, 12],'Wolves': [10, 13, 14, 15],
        }
        
        def get_squad_rank_map(year, full_map=squad_rank_map):
            index = year - 2021
            if (index < 0 or index >= len(full_map)):
                print("No Data For That Year")
                return
        
            temp_dict = {}
            for squad in full_map:
                temp_dict[squad] = full_map[squad][index]
                
            return temp_dict
        
        def apply_squad_rank(df, year):
            df = df.copy()
            rank_map_for_year = get_squad_rank_map(year)
            
            if rank_map_for_year is None:
                return df
        
            df['Squad_Rank'] = df['Unnamed: 4_level_0_Squad'].map(rank_map_for_year).apply(lambda x: 21 - x if pd.notnull(x) else 0)
            df = df.drop(columns=['Unnamed: 4_level_0_Squad'])
            
            return df
        ```
        

## 3. Analyze Data
- Correlation:
    - Players that played more matched have higher market value
    - Players with high Attack Point (Goals, Assists) have higher market value
    - Players in a Squad that ended the season with higher rank have higher market value
    - Younger players have higher market value
    - Number of Yellow cards seems to have impact on the market value, but after carefully analyzing the correlation of Number of Yellow cards to other columns, we found out that the longer a player played, more yellow card he gets.
        - This means that the number of yellow card is also representing how longer (more) a player played.

<img width="1728" height="1652" alt="image" src="https://github.com/user-attachments/assets/90af46dc-60cd-43fc-af5b-c1009d56e1ef" />


- Models:
    - Out of 8 models that we tried, Random Forest showed the best R squared score.
 
<img width="1728" height="1142" alt="image" src="https://github.com/user-attachments/assets/d6c4757d-dfe2-4fef-9709-a7463733746f" />


# Current Status and Future Recommendations

### Current Status

- Collected dataset covering four seasons, featuring performance statistics and market values for approximately 1,700 players per season.
- Processed the dataset and performed simple feature engineering.
- Provided insights on which aspect of the performance stats are strongly related to the Market Value.

### Future Recommendations

- Perform more feature engineering and obtain model with higher accuracy.
- Research another way to handle data imbalance rather than scaling.
