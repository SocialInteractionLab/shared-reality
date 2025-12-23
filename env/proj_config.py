#  config.py for shared-reality project

predict_opts = [
    "Yes",
    "No"
]

likert_dict = {
    1: "Definitely not",
    2: "Probably not",
    3: "Unsure",
    4: "Probably yes",
    5: "Definitely yes",
}

### COLOR PALETTES ###

 # blue, purple, pink; low, medium, high
match_palette = {
    "low": '#648FFF',
    "random": '#785EF0',
    "high": '#DC267F'
}
match_palette_values = list(match_palette.values())
lowhigh_palette = [match_palette_values[0], match_palette_values[2]]

# question_category_palette = {
#     "sameQ": "e79f00",
#     "sameD": "d55e00",
#     "diffD": "019e73",
# }

question_category_palette = {
    "sameQ": "#231942",
    "sameD": "#5e548e",
    "diffD": "#9f86c0",
}
question_category_palette_values = list(question_category_palette.values())

experiment_palette = {
    "no-chat": "#0fa3b1",
    "chat": "#7161ef",
}
experiment_palette_values = list(experiment_palette.values())


nQuestions = 35