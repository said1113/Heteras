import os
from dotenv import load_dotenv
import discord
from discord.ext import commands
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io
import numpy as np
import aiohttp
import asyncio # For reminders
import datetime # For reminders
import random # For quotes

# --- Load environmental variables ---
load_dotenv()

# --- AI Model paths ---
MODEL_BASE_DIR_FROM_ENV = os.getenv('MODEL_BASE_DIRECTORY')

# Environmental variables check if loaded correctly
if MODEL_BASE_DIR_FROM_ENV is None:
    print("WARNING: MODEL_BASE_DIRECTORY environment variable not set in .env file.")
    print("Falling back to a relative path. Please ensure your 'converted_keras (1)' folder is in the script's directory.")
    # Fallback to a relative path if the env var isn't found
    MODEL_BASE_DIR = 'converted_keras (1)'
else:
    MODEL_BASE_DIR = MODEL_BASE_DIR_FROM_ENV

# Full paths
MODEL_PATH = os.path.join(MODEL_BASE_DIR, 'keras_model.h5')
LABELS_PATH = os.path.join(MODEL_BASE_DIR, 'labels.txt')

# --- Discord bot initialize ---
intents = discord.Intents.default()
intents.message_content = True
intents.members = True  # Required for on_member_join/remove events
# Set the bot's name here to be used throughout the code
BOT_NAME = "Heteras"
bot = commands.Bot(command_prefix='$', intents=intents, help_command=None)

# --- PC Part Info (English Version) ---
PC_PART_INFO_EN = {
    "CPU": (
        "**CPU, stands for Central Processing Unit.**\n"
        "Also known as the **brain** of the computer, this unit is the hardware component that performs the fundamental computational operations in a computer system.\n"
        "The CPU processes instructions, handles data, and ensures all other components of the computer work in a coordinated manner."
    ),
    "GPU": (
        "**GPU, stands for Graphics Processing Unit.**\n"
        "It's specifically designed for graphics and visual computations. It's used in graphics-intensive tasks like gaming, video editing, and 3D modeling.\n"
        "It rapidly processes images, allowing you to see them on your screen."
    ),
    "RAM": (
        "**RAM, stands for Random Access Memory.**\n"
        "It's a fast type of memory that temporarily stores data the computer is actively using.\n"
        "It enables applications to run quickly and allows for multitasking. Data stored in RAM is erased when the computer is turned off."
    ),
    "MOTHERBOARD": (
        "**Motherboard, is the main circuit board that connects all the essential components of a computer.**\n"
        "It enables communication between the CPU, RAM, GPU, storage devices, and other peripherals.\n"
        "It's like the backbone of the computer."
    ),
    "SATA SSD": (
        "**SATA SSD, stands for Solid State Drive using the Serial ATA interface.**\n"
        "It offers much faster data read/write speeds compared to traditional HDDs and is more durable because it has no moving parts.\n"
        "It's commonly used as a storage solution in most desktop and laptop computers."
    ),
    "NVME SSD": (
        "**NVMe SSD, stands for Non-Volatile Memory Express interface using a Solid State Drive.**\n"
        "It offers significantly higher speeds than SATA SSDs because it plugs directly into PCIe slots, allowing for faster communication with the motherboard.\n"
        "It's ideal for high-performance applications and games."
    ),
    "HDD": (
        "**HDD, stands for Hard Disk Drive.**\n"
        "It's a traditional storage device that stores data on magnetic platters. It offers high capacities at an affordable cost.\n"
        "It's slower than SSDs and more susceptible to shocks due to its moving parts."
    ),
    "PSU": (
        "**PSU, stands for Power Supply Unit.**\n"
        "It's the hardware component that provides the correct voltage and amount of electrical power to all components of the computer.\n"
        "It's vital for the stable operation of the computer."
    ),
    "AIR COOLING": (
        "**Air Cooling, is a cooling method that uses airflow to dissipate heat from computer components, especially the CPU.**\n"
        "It typically includes a heatsink and one or more fans. It transfers heat from the component to the air."
    )
}

# --- TensorFlow Model Loading ---
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"Successfully loaded model from: {MODEL_PATH}")
except Exception as e:
    print(f"ERROR: Could not load model from {MODEL_PATH}. Please check the path and file integrity.")
    print(f"Details: {e}")
    exit()

# --- Labels Loading ---
try:
    with open(LABELS_PATH, 'r') as f:
        labels = [line.strip() for line in f]
    print(f"Successfully loaded labels from: {LABELS_PATH}")
except FileNotFoundError:
    print(f"WARNING: Labels file not found at {LABELS_PATH}. Using default labels.")
    labels = ['CPU', 'GPU', 'RAM', 'Motherboard', 'Sata SSD', 'NVMe SSD', 'HDD', 'PSU', 'Air Cooling']
except Exception as e:
    print(f"WARNING: Error reading labels file from {LABELS_PATH}: {e}. Using default labels.")
    labels = ['CPU', 'GPU', 'RAM', 'Motherboard', 'Sata SSD', 'NVMe SSD', 'HDD', 'PSU', 'Air Cooling']


# --- Global data stores for new features (for simplicity, not persistent across restarts) ---
# For Welcome/Goodbye Bot
welcome_configs = {}  # {guild_id: {'channel_id': int, 'message': str}}
goodbye_configs = {} # {guild_id: {'channel_id': int, 'message': str}}

# For Quote Bot
quotes = [] # List of strings

# --- Bot Events ---

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')
    print('Bot Ready!')

# --- Welcome & Goodbye Event Handlers ---
@bot.event
async def on_member_join(member):
    if member.guild.id in welcome_configs:
        config = welcome_configs[member.guild.id]
        channel = bot.get_channel(config['channel_id'])
        if channel:
            # Replace {user} and {server} placeholders
            welcome_message = config['message'].replace('{user}', member.mention).replace('{server}', member.guild.name)
            await channel.send(welcome_message)
        else:
            print(f"Warning: Welcome channel for guild {member.guild.name} ({member.guild.id}) not found.")

@bot.event
async def on_member_remove(member):
    if member.guild.id in goodbye_configs:
        config = goodbye_configs[member.guild.id]
        channel = bot.get_channel(config['channel_id'])
        if channel:
            # Replace {user} and {server} placeholders
            goodbye_message = config['message'].replace('{user}', member.display_name).replace('{server}', member.guild.name)
            await channel.send(goodbye_message)
        else:
            print(f"Warning: Goodbye channel for guild {member.guild.name} ({member.guild.id}) not found.")

# --- General Commands ---

@bot.command()
async def hello(ctx):
    await ctx.send(f'Hello! I am {BOT_NAME}, an AI & Moderation bot!')

@bot.command()
async def easter_egg(ctx):
    await ctx.send(f'Hey {BOT_NAME}, no code today. Try again later!')

# --- Help Command (Updated for Heteras and new features) ---
@bot.command()
async def help(ctx):
    help_message = (
        f'The following commands and syntax are personalized for the **{BOT_NAME}** Discord bot!\n\n'
        f'**General Commands:**\n'
        f'`$hello` - Bot introduces itself.\n'
        f'`$easter_egg` - Sends a small surprise message.\n'
        f'`$info <part name>` - Gives information about a specific PC part.\n\n'
        f'**PC Part Prediction Commands:**\n'
        f'`$predict` - Predicts the PC part in a photo you upload. (Attach the photo with the command)\n'
        f'`$net_predict <photo_url>` - Predicts the PC part in a photo from a specified URL.\n\n'
        f'**Moderation Commands (Admins only):**\n'
        f'`$kick <@user> [reason]` - Kicks a user from the server.\n'
        f'`$ban <@user> [reason]` - Bans a user from the server.\n'
        f'`$mute <@user> [duration (e.g., 10m, 2h)] [reason]` - Mutes a user for a specified duration.\n'
        f'`$clear <number>` - Deletes a specified number of messages from the channel.\n\n'
        f'**Welcome & Goodbye Commands (Admins only):**\n'
        f'`$setwelcome <#channel> [message]` - Sets the welcome message and channel for new members. (You can use `{{user}}` and `{{server}}` placeholders.)\n'
        f'`$setgoodbye <#channel> [message]` - Sets the goodbye message and channel for departing members. (You can use `{{user}}` and `{{server}}` placeholders.)\n'
        f'`$showwelcome` - Shows the currently set welcome message and channel.\n'
        f'`$showgoodbye` - Shows the currently set goodbye message and channel.\n\n'
        f'**Poll Commands:**\n'
        f'`$poll "<question>" "<option1>" "<option2>" ...` - Starts a new poll. Enter at least 2, and up to 9 options.\n\n'
        f'**Reminder Commands:**\n'
        f'`$remindme <time (e.g., 5m, 2h, 1d)> <message>` - Sends you a private reminder after the specified time.\n\n'
        f'**Quote Commands:**\n'
        f'`$addquote [message]` - Adds a quote from a replied message or the provided text.\n'
        f'`$quote` - Shows a random quote.\n\n'
        f'**Usage Examples:**\n'
        f'`$predict` (and attach an image)\n'
        f'`$net_predict https://example.com/some_pc_part.jpg`\n'
        f'`$info CPU`\n'
        f'`$setwelcome #general Welcome {user}! Thanks for joining {server}.`\n'
        f'`$poll "What\'s your favorite PC part?" "CPU" "GPU" "RAM"`\n'
        f'`$remindme 30m Meeting starts soon!`\n'
        f'`$addquote This is a hilarious quote!`\n\n'
        f'**Please be careful when using! Ensure the object is alone in the provided photo. Otherwise, the chances of a wrong prediction are higher!**\n\n'
        f'What I can predict:\n'
        f'0 CPU (Central Processing Unit)\n'
        f'1 GPU (Graphics Processing Unit)\n'
        f'2 RAM (Random Access Memory)\n'
        f'3 Motherboard\n'
        f'4 Sata SSD\n'
        f'5 NVMe SSD\n'
        f'6 HDD\n'
        f'7 PSU (Power Supply)\n'
        f'8 Air Cooling\n\n'
        f'**This Discord bot uses AI for predictions and may give an incorrect answer! Always double-check!**'
    )
    await ctx.send(help_message)


# --- Image Processing Functions ---
def preprocess_image(image_bytes, target_size):
    img = Image.open(io.BytesIO(image_bytes)).resize(target_size)
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

async def predict_image(image_bytes):
    try:
        img_array = preprocess_image(image_bytes, (model.input_shape[1], model.input_shape[2]))
        predictions = model.predict(img_array)
        decoded_predictions = tf.nn.softmax(predictions).numpy()
        top_prediction_index = np.argmax(decoded_predictions[0])
        confidence = decoded_predictions[0][top_prediction_index]
        predicted_label = labels[top_prediction_index]
        return predicted_label, confidence

    except Exception as e:
        print(f"Couldn't process the photo: {e}")
        return None, None

# --- PC Part Prediction Commands ---

@bot.command()
async def predict(ctx):
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            if attachment.content_type and attachment.content_type.startswith('image/'):
                image_bytes = await attachment.read()
                prediction, confidence = await predict_image(image_bytes)

                if prediction:
                    await ctx.send(f"This is a **{prediction}** and I say this with **{confidence:.2f}%** confidence.")
                else:
                    await ctx.send("Couldn't process the photo.")
            else:
                await ctx.send("Please make sure you attached a photo.")
    else:
        await ctx.send("To predict, please provide a photo.")

@bot.command()
async def net_predict(ctx, image_url: str):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as resp:
                if resp.status == 200:
                    image_bytes = await resp.read()
                    prediction, confidence = await predict_image(image_bytes)
                    if prediction:
                        await ctx.send(f"The photo in the URL looks like a **{prediction}**. And I say this with a **{confidence:.2f}%** confidence.")
                    else:
                        await ctx.send("Couldn't process the photo in the URL.")
                else:
                    await ctx.send(f"Couldn't download the photo from the URL. Status code: {resp.status}")
    except aiohttp.ClientConnectorError as e:
        await ctx.send(f"Couldn't connect to the URL given: {e}. Please make sure the URL is valid.")
    except Exception as e:
        await ctx.send(f"A problem occurred: {e}")

# --- PC Part Info Command ---
@bot.command()
async def info(ctx, *, part_name: str):
    part_name_upper = part_name.upper()

    # Improved logic for common variations
    if "AIR COOLING" in part_name_upper or "HAVA SOĞUTMA" in part_name_upper:
        requested_part = "AIR COOLING"
    elif "SATA SSD" in part_name_upper:
        requested_part = "SATA SSD"
    elif "NVME SSD" in part_name_upper:
        requested_part = "NVME SSD"
    elif "HDD" in part_name_upper:
        requested_part = "HDD"
    elif "PSU" in part_name_upper:
        requested_part = "PSU"
    elif "RAM" in part_name_upper:
        requested_part = "RAM"
    elif "GPU" in part_name_upper or "EKRAN KARTI" in part_name_upper:
        requested_part = "GPU"
    elif "CPU" in part_name_upper or "İŞLEMCİ" in part_name_upper:
        requested_part = "CPU"
    elif "MOTHERBOARD" in part_name_upper or "ANAKART" in part_name_upper:
        requested_part = "MOTHERBOARD"
    else:
        requested_part = part_name_upper

    info_message = PC_PART_INFO_EN.get(requested_part) # Changed to PC_PART_INFO_EN

    if info_message:
        await ctx.send(info_message)
    else:
        await ctx.send(
            f"Sorry, couldn't find info about **{part_name}**. "
            "Parts I can provide info about for now: CPU, GPU, RAM, Motherboard, SATA SSD, NVMe SSD, HDD, PSU, Air Cooling. "
            "Please try to use the parts listed above."
        )

# --- New Feature: Welcome & Goodbye Commands ---

@bot.command()
@commands.has_permissions(manage_channels=True)
async def setwelcome(ctx, channel: discord.TextChannel, *, message: str):
    """Sets the welcome message and channel for new members."""
    welcome_configs[ctx.guild.id] = {'channel_id': channel.id, 'message': message}
    await ctx.send(f"Welcome message set. New members will receive this message in {channel.mention}:\n`{message}`")
    await ctx.send("You can use `{user}` for the user's name and `{server}` for the server's name.")

@setwelcome.error
async def setwelcome_error(ctx, error):
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("Please specify a channel and a message. Example: `$setwelcome #general Welcome {user}!`")
    elif isinstance(error, commands.BadArgument):
        await ctx.send("Invalid channel. Please tag a valid text channel.")
    elif isinstance(error, commands.MissingPermissions):
        await ctx.send("You need 'Manage Channels' permission to use this command.")
    else:
        await ctx.send(f"An error occurred: {error}")

@bot.command()
@commands.has_permissions(manage_channels=True)
async def setgoodbye(ctx, channel: discord.TextChannel, *, message: str):
    """Sets the goodbye message and channel for departing members."""
    goodbye_configs[ctx.guild.id] = {'channel_id': channel.id, 'message': message}
    await ctx.send(f"Goodbye message set. Departing members will receive this message in {channel.mention}:\n`{message}`")
    await ctx.send("You can use `{user}` for the user's name and `{server}` for the server's name.")

@setgoodbye.error
async def setgoodbye_error(ctx, error):
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("Please specify a channel and a message. Example: `$setgoodbye #general Goodbye {user}.`")
    elif isinstance(error, commands.BadArgument):
        await ctx.send("Invalid channel. Please tag a valid text channel.")
    elif isinstance(error, commands.MissingPermissions):
        await ctx.send("You need 'Manage Channels' permission to use this command.")
    else:
        await ctx.send(f"An error occurred: {error}")

@bot.command()
async def showwelcome(ctx):
    """Shows the current welcome message and channel."""
    if ctx.guild.id in welcome_configs:
        config = welcome_configs[ctx.guild.id]
        channel = bot.get_channel(config['channel_id'])
        if channel:
            await ctx.send(f"Configured welcome channel: {channel.mention}\nConfigured welcome message: `{config['message']}`")
        else:
            await ctx.send("The configured welcome channel was not found or has been deleted.")
    else:
        await ctx.send("No welcome message has been set for this server yet.")

@bot.command()
async def showgoodbye(ctx):
    """Shows the current goodbye message and channel."""
    if ctx.guild.id in goodbye_configs:
        config = goodbye_configs[ctx.guild.id]
        channel = bot.get_channel(config['channel_id'])
        if channel:
            await ctx.send(f"Configured goodbye channel: {channel.mention}\nConfigured goodbye message: `{config['message']}`")
        else:
            await ctx.send("The configured goodbye channel was not found or has been deleted.")
    else:
        await ctx.send("No goodbye message has been set for this server yet.")

# --- New Feature: Simple Moderation Commands ---

@bot.command()
@commands.has_permissions(kick_members=True)
async def kick(ctx, member: discord.Member, *, reason=None):
    """Kicks a member from the server."""
    await member.kick(reason=reason)
    await ctx.send(f'{member.display_name} has been kicked from the server. Reason: {reason if reason else "Not specified."}')

@kick.error
async def kick_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("You need 'Kick Members' permission to use this command.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("Please specify the member to kick. Example: `$kick @user spamming`")
    elif isinstance(error, commands.BadArgument):
        await ctx.send("Invalid user. Please tag a valid user.")
    else:
        await ctx.send(f"An error occurred: {error}")


@bot.command()
@commands.has_permissions(ban_members=True)
async def ban(ctx, member: discord.Member, *, reason=None):
    """Bans a member from the server."""
    await member.ban(reason=reason)
    await ctx.send(f'{member.display_name} has been banned from the server. Reason: {reason if reason else "Not specified."}')

@ban.error
async def ban_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("You need 'Ban Members' permission to use this command.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("Please specify the member to ban. Example: `$ban @user violating rules`")
    elif isinstance(error, commands.BadArgument):
        await ctx.send("Invalid user. Please tag a valid user.")
    else:
        await ctx.send(f"An error occurred: {error}")

@bot.command()
@commands.has_permissions(manage_roles=True)
async def mute(ctx, member: discord.Member, duration: str = None, *, reason=None):
    """Mutes a member for a specified duration."""
    if not duration:
        await ctx.send("Please specify a duration (e.g., 10m, 2h, 1d).")
        return

    # Parse duration
    time_units = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
    unit = duration[-1].lower()
    try:
        amount = int(duration[:-1])
        seconds = amount * time_units.get(unit, 0)
        if seconds == 0:
            raise ValueError
    except (ValueError, KeyError):
        await ctx.send("Invalid duration format. Please use a format like '10m', '2h', '1d'.")
        return

    # Find or create a 'Muted' role
    muted_role = discord.utils.get(ctx.guild.roles, name="Muted")
    if not muted_role:
        await ctx.send("Muted role not found, attempting to create it...")
        try:
            # Create role with permissions to prevent sending messages, adding reactions, speaking
            muted_role = await ctx.guild.create_role(name="Muted", permissions=discord.Permissions(send_messages=False, add_reactions=False, speak=False))
            # Set permissions for the Muted role in all channels
            for channel in ctx.guild.channels:
                await channel.set_permissions(muted_role, send_messages=False, add_reactions=False, speak=False)
            await ctx.send("Muted role successfully created and channel permissions set.")
        except discord.Forbidden:
            await ctx.send("I do not have sufficient permissions to create the Muted role.")
            return

    if muted_role in member.roles:
        await ctx.send(f"{member.display_name} is already muted.")
        return

    await member.add_roles(muted_role, reason=reason)
    await ctx.send(f'{member.display_name} has been muted. Duration: {duration}, Reason: {reason if reason else "Not specified."}')

    await asyncio.sleep(seconds)
    await member.remove_roles(muted_role, reason="Mute duration expired.")
    await ctx.send(f'{member.display_name}\'s mute has been lifted.')

@mute.error
async def mute_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("You need 'Manage Roles' permission to use this command.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("Please specify the member to mute, duration, and optionally a reason. Example: `$mute @user 30m spam`")
    elif isinstance(error, commands.BadArgument):
        await ctx.send("Invalid user. Please tag a valid user.")
    else:
        await ctx.send(f"An error occurred: {error}")

@bot.command()
@commands.has_permissions(manage_messages=True)
async def clear(ctx, amount: int):
    """Clears a specified number of messages from the channel."""
    if amount <= 0:
        await ctx.send("Please enter a positive number.")
        return
    if amount > 100:
        await ctx.send("You can only delete up to 100 messages at a time.")
        amount = 100 # Cap at 100 to prevent abuse or API rate limits

    await ctx.channel.purge(limit=amount + 1)  # +1 to also delete the command message itself
    await ctx.send(f'{amount} messages deleted.', delete_after=5) # Delete this message after 5 seconds

@clear.error
async def clear_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("You need 'Manage Messages' permission to use this command.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("Please specify the number of messages to delete. Example: `$clear 10`")
    elif isinstance(error, commands.BadArgument):
        await ctx.send("Invalid number. Please enter an integer.")
    else:
        await ctx.send(f"An error occurred: {error}")

# --- New Feature: Poll & Voting Bot ---

@bot.command()
async def poll(ctx, question: str, *options: str):
    """Creates a poll with up to 9 options.
    Example: $poll "What's your favorite color?" "Red" "Blue" "Green"
    """
    if len(options) < 2:
        await ctx.send("Please provide at least two options.")
        return
    if len(options) > 9:
        await ctx.send("You can use a maximum of 9 options.")
        return

    emoji_numbers = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"]
    description = []
    for i, option in enumerate(options):
        description.append(f"{emoji_numbers[i]} {option}")

    embed = discord.Embed(
        title=question,
        description="\n".join(description),
        color=discord.Color.blue()
    )
    embed.set_footer(text=f"Poll started by: {ctx.author.display_name}")

    poll_message = await ctx.send(embed=embed)
    for i in range(len(options)):
        await poll_message.add_reaction(emoji_numbers[i])

# --- New Feature: Reminder Bot ---

@bot.command()
async def remindme(ctx, time_str: str, *, reminder_message: str):
    """Sets a personal reminder. Example: $remindme 30m Buy groceries"""
    time_units = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
    unit = time_str[-1].lower()
    try:
        amount = int(time_str[:-1])
        seconds = amount * time_units.get(unit, 0)
        if seconds == 0:
            raise ValueError
    except (ValueError, KeyError):
        await ctx.send("Invalid duration format. Please use a format like '10m', '2h', '1d'.")
        return

    if seconds > 86400 * 7: # Max 7 days reminder
        await ctx.send("You can set a reminder for a maximum of 7 days.")
        return

    await ctx.send(f"Reminder set! I will remind you in your DMs in **{amount}{unit}**: `{reminder_message}`")
    await asyncio.sleep(seconds)
    try:
        await ctx.author.send(f"Reminder: **{reminder_message}**")
    except discord.Forbidden:
        await ctx.send(f"{ctx.author.mention}, I couldn't send you a private message. Please ensure your DMs are open! Reminder: **{reminder_message}**")

@remindme.error
async def remindme_error(ctx, error):
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("Please specify a duration and a reminder message. Example: `$remindme 1h Project deadline`")
    else:
        await ctx.send(f"An error occurred: {error}")

# --- New Feature: Quote Bot ---

@bot.command()
async def addquote(ctx, *, message_content: str = None):
    """Adds a quote from a replied message or directly from the command.
    Example: $addquote This is a new quote!
    Or reply to a message and use $addquote
    """
    quote_text = None
    if ctx.message.reference and ctx.message.reference.message_id:
        try:
            # Fetch the message being replied to
            replied_message = await ctx.channel.fetch_message(ctx.message.reference.message_id)
            quote_text = replied_message.content
            if not quote_text:
                await ctx.send("The replied message contains no text content.")
                return
            quote_text += f" - {replied_message.author.display_name}"
        except discord.NotFound:
            await ctx.send("The replied message was not found.")
            return
        except Exception as e:
            await ctx.send(f"An error occurred while fetching the replied message: {e}")
            return
    elif message_content:
        quote_text = message_content + f" - {ctx.author.display_name}"
    else:
        await ctx.send("Please reply to a message or provide text to add a quote. Example: `$addquote This was hilarious!`")
        return

    if quote_text:
        quotes.append(quote_text)
        await ctx.send("Quote successfully added!")

@bot.command()
async def quote(ctx):
    """Retrieves a random quote."""
    if quotes:
        random_quote = random.choice(quotes)
        await ctx.send(f"**Quote:** \"{random_quote}\"")
    else:
        await ctx.send("No quotes have been added yet. Use `$addquote` to add one!")

# --- Run the bot ---
DISCORD_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
if DISCORD_TOKEN is None:
    print("ERROR: DISCORD_BOT_TOKEN environment variable not set.")
    print("Please create a .env file in the same directory as your script with DISCORD_BOT_TOKEN=YOUR_TOKEN_HERE")
    exit()

bot.run(DISCORD_TOKEN)