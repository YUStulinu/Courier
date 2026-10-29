"""What the voices say: the wake phrase in several pronunciations, phrases that sound like it, and
ordinary speech."""

# "Hey" and "Courier" as IPA (espeak's style, which the Piper voices read). People say "courier" as
# CUR-ee-er or COOR-ee-er, with or without the r in the middle.
HEY = ["hˈeɪ", "hˌeɪ", "heɪ", "hˈɛɪ"]
COURIER = ["kˈɜːiːɚ", "kˈʊɹiɚ", "kˈʊɹiə", "kˈɜːɹiɚ", "kˈʌɹiɚ", "kˈʊɹiːɚ"]
ENDINGS = ["", ".", "!", "?", ","]


def wake_phrase(rng) -> str:
    hey = HEY[rng.integers(len(HEY))]
    sep = ", " if rng.random() < 0.25 else " "
    return f"[[{hey}]]{sep}[[{COURIER[rng.integers(len(COURIER))]}]]{ENDINGS[rng.integers(len(ENDINGS))]}"


# Things that sound like the wake phrase, or contain half of it. None of them may wake the assistant.
CONFUSABLES = [
    "hey courtney", "hey cory", "hey career", "hey curious", "hey carrier", "hey kerry", "hey curry",
    "hey korea", "hey jeweler", "hey warrior", "hey terrier", "hey barrier", "hey cooper", "hey courtyard",
    "hey furrier", "hey worrier", "hey, cool", "hey there", "hey you", "hey computer", "hey google",
    "hey siri", "hey cortana", "okay google", "alexa", "hey, could you", "hey, curious george",
    "the courier", "a courier", "the courier is late", "send it by courier", "courier service",
    "call the courier", "courier", "courier, please", "a courier arrived with a package for you",
    "they hired a courier to deliver the documents", "the courier left the parcel at the door",
    "hey, carrie", "hey, query", "hey, quarry", "hey, career day", "heavy courier", "hey, curtain",
    "hey courts", "hey, hurry", "hey, worry", "hey, fury", "hey, jury", "hey, mercury",
]

# Requests said to someone else, or to the assistant without waking it.
COMMANDS = [
    "set a timer for ten minutes", "what's the weather like tomorrow", "play some jazz",
    "remind me to call my mother at six", "turn the volume down", "what's on my calendar today",
    "find my notes about the boiler", "open the browser", "pause the music", "next song please",
    "how long until the pasta is done", "add a meeting with Anna on Friday at three",
    "what time is it", "lock the computer", "stop the timer", "is it going to rain this afternoon",
    "can you read me my reminders", "where did I put the tax documents", "play the album I listened to yesterday",
    "turn it up a little", "skip this one", "what's twelve times fifteen", "tell me a joke",
    "good morning", "thank you", "never mind", "cancel that", "that's all for now",
    "how do I make pancakes", "who wrote pride and prejudice", "set an alarm for seven thirty",
    "what's the capital of Portugal", "how far is the moon", "remind me to water the plants tomorrow morning",
    "start a five minute timer for the tea", "what did I schedule for next week", "open my downloads folder",
    "mute the sound", "resume the music", "what's playing right now", "search my documents for invoice",
]

# Short filler to put around the wake phrase.
LEAD_INS = ["um", "okay", "so", "right", "and", "oh", "well", "uh", "alright"]
