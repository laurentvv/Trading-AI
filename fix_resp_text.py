
with open("src/t212_executor.py", "r") as f:
    content = f.read()

# Fix buy response error handling
old_buy_error = "logger.error(f\"❌ Échec de l'achat : {resp.text}\")"
new_buy_error = """if resp is None:
                logger.error("❌ Échec de l'achat : réseau (pas de réponse de l'API)")
            else:
                logger.error(f"❌ Échec de l'achat : {resp.text}")"""

if old_buy_error in content:
    content = content.replace(old_buy_error, new_buy_error)

# Fix sell response error handling
old_sell_error = "logger.error(f\"❌ Erreur lors de la vente : {sell_resp.text}\")"
new_sell_error = """if sell_resp is None:
                logger.error("❌ Erreur lors de la vente : réseau (pas de réponse de l'API)")
            else:
                logger.error(f"❌ Erreur lors de la vente : {sell_resp.text}")"""

if old_sell_error in content:
    content = content.replace(old_sell_error, new_sell_error)

with open("src/t212_executor.py", "w") as f:
    f.write(content)
