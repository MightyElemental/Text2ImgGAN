from discord_webhook import DiscordWebhook, DiscordEmbed
import random

color = random.randint(0,0xFFFFFF)

webhook = None

def init_webook(path:str):
    global webhook
    with open(path, "r") as f:
        webhook = f.readline()

def submit(title:str, text:str=None, footer:str=None, img_path=None):
    wh = DiscordWebhook(url=webhook)

    embed = DiscordEmbed(title=title, description=text, color=color)
    if footer is not None:
        embed.set_footer(text=footer)
        embed.set_timestamp()

    wh.add_embed(embed)

    if img_path is not None:
        with open(img_path, "rb") as f:
            wh.add_file(file=f.read(), filename=img_path)

    response = wh.execute()
    if response.status_code != 200:
        print("Failed to send msg:", response.text)