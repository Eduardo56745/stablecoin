import ccxt
import pandas as pd
import numpy as np
from stable_baselines3 import DQN
import time
import os
from typing import Dict, Any

# =======================================================
# 1. FUNCIONES DE INDICADORES TÉCNICOS
# =======================================================


def calculate_rsi(series, period=14):
    """Calcula el Índice de Fuerza Relativa (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd_hist(series, fast=12, slow=26, signal=9):
    """Calcula el Histograma MACD."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

# =======================================================
# 2. CONFIGURACIÓN DEL BOT Y CONEXIÓN
# =======================================================


# ⚠️ VALORES DE CONFIGURACIÓN
MODEL_PATH = "dqn_yield_farming_model_1760403129.zip"
EXCHANGE_ID = 'binance'
SYMBOL = 'ETH/USDT'
TIMEFRAME = '1h'

# ✅ CREDENCIALES DE BINANCE (USANDO TUS VALORES)
API_KEY = 'YVTqmWyuMpI1aUB2bfNKQ16peny2tC0aAu6kKhdA6wkgwQSGcWUiZPNV17eoT2nS'
SECRET = '73YzB64IPFismn1T5JokjClgGOxMIcW3ByGMUVrSf6V4nXsaBNsClz3bxvQelPm4'

# Conectar al exchange
exchange = getattr(ccxt, EXCHANGE_ID)({
    'apiKey': API_KEY,
    'secret': SECRET,
    'enableRateLimit': True,
})

# Cargar el modelo ML entrenado
try:
    model = DQN.load(MODEL_PATH)
    print("✅ Modelo DQN cargado con éxito.")
except Exception as e:
    # Esto ocurre si el archivo ZIP no está en la carpeta correcta
    print(f"❌ Error al cargar el modelo: {e}")
    exit()

# =======================================================
# 3. FUNCIONES DE EJECUCIÓN DEL TRADE (CORRECCIÓN FINAL)
# =======================================================


def execute_action(action: int, current_price: float, current_balance: dict, asset_quantity: float):
    """Ejecuta la acción (BUY, SELL, HOLD) en el exchange, manejando la precisión mínima y el buffer de comisión."""

    # Mínimos requeridos por Binance
    MIN_ETH_AMOUNT = 0.00011
    MIN_USD_ORDER = 10.0
    # 💥 CORRECCIÓN FINAL: Usar solo 99.0% del USDT (margen para fees)
    BUY_BUFFER_RATIO = 0.990

    live_position = 1 if asset_quantity > 0 else 0

    if action == 1:  # BUY
        if live_position == 0:
            print(f"🟢 DECISIÓN ML: COMPRAR ETH en ${current_price:,.2f}")

            usd_balance = current_balance['USDT']

            if usd_balance < MIN_USD_ORDER:
                print(
                    f"   [AVISO] Balance insuficiente ({usd_balance:.2f} USDT) para operar (Min: ${MIN_USD_ORDER}).")
                return current_balance['ETH']

            # 💥 APLICACIÓN DEL BUFFER
            usd_to_spend = usd_balance * BUY_BUFFER_RATIO
            amount_to_buy = usd_to_spend / current_price

            amount_to_buy = exchange.amount_to_precision(SYMBOL, amount_to_buy)

            # 🚀 ORDEN DE COMPRA REAL
            exchange.create_market_buy_order(SYMBOL, amount_to_buy)
            print(
                f"   [ORDEN EJECUTADA] Compra de {amount_to_buy} ETH usando {usd_to_spend:.2f} USDT")

            return amount_to_buy

    elif action == 2:  # SELL
        if live_position == 1:
            print(f"🔴 DECISIÓN ML: VENDER ETH en ${current_price:,.2f}")

            amount_to_sell = exchange.amount_to_precision(
                SYMBOL, asset_quantity)

            # Verifica el mínimo antes de vender
            if float(amount_to_sell) < MIN_ETH_AMOUNT:  # Usamos float() para seguridad
                print(
                    f"   [AVISO] Cantidad de ETH ({amount_to_sell}) por debajo del mínimo ({MIN_ETH_AMOUNT}). NO SE EJECUTA LA VENTA.")
                return asset_quantity

            # 🚀 ORDEN DE VENTA REAL
            exchange.create_market_sell_order(SYMBOL, amount_to_sell)
            print(f"   [ORDEN EJECUTADA] Venta de {amount_to_sell} ETH")

            return 0.0

    else:  # action == 0 (HOLD)
        print("⚪ DECISIÓN ML: MANTENER POSICIÓN")
        return asset_quantity

# =======================================================
# 4. CICLO DE EJECUCIÓN PRINCIPAL
# =======================================================


def run_trading_cycle():
    """Función principal que se ejecuta cada hora."""

    MIN_USD_ORDER = 10.0

    print(f"\n--- Iniciando Ciclo de Trading: {pd.Timestamp.now()} ---")

    # 4.1 OBTENER BALANCE Y ESTADO ACTUAL
    try:
        balance = exchange.fetch_balance()

        # 🚨 CORRECCIÓN DE TIPO CRÍTICA: Forzar la conversión a float
        balance_usdt_free = balance['USDT']['free'] if 'USDT' in balance and balance['USDT']['free'] is not None else 0.0
        balance_eth_free = balance['ETH']['free'] if 'ETH' in balance and balance['ETH']['free'] is not None else 0.0

        current_balance = {
            'USDT': float(balance_usdt_free),
            'ETH': float(balance_eth_free)
        }

        ticker = exchange.fetch_ticker(SYMBOL)
        current_price = ticker['last']

        # Ignorar el 'polvo' de ETH
        eth_value_usd = current_balance['ETH'] * current_price

        if eth_value_usd < MIN_USD_ORDER:
            asset_quantity = 0.0
            live_position = 0
            current_balance['USDT'] += eth_value_usd
        else:
            asset_quantity = current_balance['ETH']
            live_position = 1

        capital = current_balance['USDT'] + (asset_quantity * current_price)

        print(
            f"💰 Balance Actual (Total): ${capital:,.2f} USD | Posición: {'ETH' if live_position == 1 else 'USDT'}")

    except Exception as e:
        print(
            f"❌ Error al obtener balance o precio: {e}. Asegúrate de que las claves y permisos sean correctos.")
        return

    # 4.2 OBTENER DATOS HISTÓRICOS PARA CALCULAR INDICADORES
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=30)
        df_live = pd.DataFrame(
            ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        price_series = df_live['close']

        # 4.3 CALCULAR EL ESTADO (OBSERVACIÓN)
        df_live['RSI'] = calculate_rsi(price_series)
        df_live['MACD_Hist'] = calculate_macd_hist(price_series)

        latest_data = df_live.iloc[-1]

        price_change_norm = price_series.pct_change(
        ).iloc[-1] if len(price_series) >= 2 else 0.0

        current_obs = np.array([
            latest_data['RSI'],
            latest_data['MACD_Hist'],
            price_change_norm,
            live_position
        ], dtype=np.float32)

    except Exception as e:
        print(f"❌ Error al obtener OHLCV o calcular ITs: {e}")
        return

    # 4.4 PREDECIR Y EJECUTAR

    current_obs_reshaped = current_obs.reshape(1, -1)
    action_array, _states = model.predict(
        current_obs_reshaped, deterministic=True)
    action = int(action_array[0])

    execute_action(action, latest_data['close'],
                   current_balance, asset_quantity)

# =======================================================
# 5. BUCLE INFINITO (EJECUCIÓN DEL BOT)
# =======================================================


while True:
    try:
        run_trading_cycle()

        print("\n--- FIN DE CICLO. Esperando 60 minutos ---")
        time.sleep(3600)

    except Exception as e:
        print(
            f"🔴 ERROR FATAL EN EL BUCLE PRINCIPAL: {e}. Reiniciando en 5 minutos.")
        time.sleep(300)
