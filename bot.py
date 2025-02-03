import requests
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json
import schedule
import time

# Load config
def load_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config

# Fetch data from DexScreener
def fetch_dexscreener_data():
    url = "https://api.dexscreener.com/latest/dex/tokens/"  # Example API endpoint
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Failed to fetch data from DexScreener")
        return None

# Check RugCheck status
def check_rugcheck(token_address, config):
    rugcheck_api_key = config.get('rugcheck_api_key')
    if not rugcheck_api_key:
        print("RugCheck API key not found in config.")
        return False
    
    url = f"https://api.rugcheck.xyz/v1/tokens/{token_address}"
    headers = {"Authorization": f"Bearer {rugcheck_api_key}"}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get('status') == "Good"
        else:
            print(f"Failed to fetch data from RugCheck: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error checking RugCheck: {e}")
        return False

# Check for bundled supply
def is_supply_bundled(token_address, threshold=0.5):
    url = f"https://api.etherscan.io/api?module=account&action=tokenbalance&contractaddress={token_address}&tag=latest&apikey=YourApiKey"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        holders = data.get('result', [])
        if holders:
            total_supply = sum(float(h['balance']) for h in holders)
            largest_holder = max(float(h['balance']) for h in holders)
            if largest_holder / total_supply > threshold:
                return True  # Supply is bundled
    return False  # Supply is not bundled

# Parse token data and apply filters/blacklists
def parse_token_data(data, config):
    tokens = []
    for token in data['pairs']:
        token_address = token['baseToken']['address']
        token_symbol = token['baseToken']['symbol']
        token_dev = token['baseToken'].get('developer', 'unknown')
        
        # Check if token or dev is blacklisted
        if (token_symbol in config['blacklist']['coins'] or
            token['baseToken']['name'] in config['blacklist']['coins'] or
            token_dev in config['blacklist']['devs']):
            continue  # Skip blacklisted tokens/devs
        
        # Check RugCheck status
        if not check_rugcheck(token_address, config):
            print(f"Skipping {token_symbol} (Bad status on RugCheck).")
            # Add token and dev to blacklist
            config['blacklist']['coins'].append(token_symbol)
            config['blacklist']['devs'].append(token_dev)
            continue
        
        # Check for bundled supply
        if is_supply_bundled(token_address):
            print(f"Skipping {token_symbol} (Bundled supply detected).")
            # Add token and dev to blacklist
            config['blacklist']['coins'].append(token_symbol)
            config['blacklist']['devs'].append(token_dev)
            continue
        
        # Apply filters
        if (token['volume']['h24'] < config['filters']['min_volume'] or
            float(token['priceUsd']) > config['filters']['max_price'] or
            token['liquidity']['usd'] < config['filters']['min_liquidity']):
            continue  # Skip tokens that don't meet filter criteria
        
        # Check for fake volume
        token_info = {
            'name': token['baseToken']['name'],
            'symbol': token_symbol,
            'price': token['priceUsd'],
            'volume': token['volume']['h24'],
            'liquidity': token['liquidity']['usd'],
            'market_cap': token['fdv'],
            'chain': token['chainId'],
            'exchange': token['dexId'],
            'developer': token_dev
        }
        if detect_fake_volume(token_info, config):
            print(f"Skipping {token_symbol} due to fake volume.")
            continue  # Skip tokens with fake volume
        
        tokens.append(token_info)
    return pd.DataFrame(tokens)

# Save data to database
def save_to_db(df, db_name='crypto_data.db'):
    engine = create_engine(f'sqlite:///{db_name}')
    df.to_sql('tokens', engine, if_exists='append', index=False)
    print("Data saved to database.")

# Analyze data
def analyze_data(db_name='crypto_data.db'):
    engine = create_engine(f'sqlite:///{db_name}')
    df = pd.read_sql('tokens', engine)
    
    # Example: Predict if a token will pump based on historical data
    df['pumped'] = df['volume'].apply(lambda x: 1 if x > 1_000_000 else 0)  # Example label
    X = df[['price', 'volume', 'liquidity', 'market_cap']]
    y = df['pumped']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")

# Send Telegram notification
def send_telegram_message(message, config):
    bot_token = config['telegram']['bot_token']
    chat_id = config['telegram']['chat_id']
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Telegram notification sent.")
        else:
            print(f"Failed to send Telegram notification: {response.status_code}")
    except Exception as e:
        print(f"Error sending Telegram notification: {e}")

# Execute trade via BonkBot
def execute_trade_via_bonkbot(token_symbol, action, config):
    bonkbot_api_key = config['bonkbot']['api_key']
    url = f"https://api.bonkbot.com/v1/trade"
    headers = {"Authorization": f"Bearer {bonkbot_api_key}"}
    payload = {
        "symbol": token_symbol,
        "action": action  # "buy" or "sell"
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            print(f"{action.capitalize()} order for {token_symbol} executed successfully.")
            return True
        else:
            print(f"Failed to execute {action} order: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error executing trade: {e}")
        return False

# Run the bot
def run_bot():
    config = load_config()
    data = fetch_dexscreener_data()
    if data:
        df = parse_token_data(data, config)
        if not df.empty:
            save_to_db(df)
            analyze_data()
            
            # Example: Execute a trade for the first token in the list
            token_symbol = df.iloc[0]['symbol']
            if execute_trade_via_bonkbot(token_symbol, "buy", config):
                send_telegram_message(f"Buy order executed for {token_symbol}.", config)
        else:
            print("No tokens matched the filters.")

# Schedule the bot to run every 5 minutes
schedule.every(5).minutes.do(run_bot)

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(1)