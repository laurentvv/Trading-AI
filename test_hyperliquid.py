from hyperliquid.info import Info
from hyperliquid.utils import constants
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HyperliquidTest")

def test_hyperliquid_oil():
    # Setup for Mainnet
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    
    logger.info("Fetching data from Hyperliquid for OIL assets...")
    try:
        # Check Perps (Native and HIP-3)
        logger.info("Checking Perpetuals metadata and context...")
        meta_data = info.meta_and_asset_ctxs()
        
        if meta_data and len(meta_data) == 2:
            universe_dict, contexts = meta_data
            universe = universe_dict.get('universe', [])
            
            for i, asset_meta in enumerate(universe):
                name = asset_meta.get('name')
                if name and ("OIL" in name.upper() or "WTI" in name.upper()):
                    # Match context by index
                    if i < len(contexts):
                        ctx = contexts[i]
                        print(f"\n--- OIL ASSET FOUND ---")
                        print(f"Name: {name}")
                        print(f"  Mark Price: {ctx.get('markPx')}")
                        funding = float(ctx.get('funding', 0)) * 100
                        print(f"  Hourly Funding: {funding:.6f}%")
                        print(f"  Open Interest: {ctx.get('openInterest')}")
                        print(f"  Daily Volume: {ctx.get('dayNtlVlm')}")
                    
        # Also check all mids just in case
        all_mids = info.all_mids()
        for name, price in all_mids.items():
            if "OIL" in name.upper() or "WTI" in name.upper():
                if "km:USOIL" in name or "flx:OIL" in name:
                    print(f"Mid Price check: {name} = {price}")

    except Exception as e:
        logger.error(f"Error fetching data: {e}")

if __name__ == "__main__":
    test_hyperliquid_oil()
