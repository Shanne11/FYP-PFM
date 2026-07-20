"""
Simple Rule-Based Transaction Categorisation
"""

rules = {

    "electricity": "Utilities",
    "water": "Utilities",
    "internet": "Utilities",

    "movie": "Entertainment",
    "cinema": "Entertainment",
    "netflix": "Entertainment",

    "book": "Education",
    "school": "Education",
    "tuition": "Education",

    "restaurant": "Food",
    "food": "Food",
    "coffee": "Food",
    "lunch": "Food",
    "dinner": "Food",

    "uber": "Travel",
    "grab": "Travel",
    "taxi": "Travel",
    "bus": "Travel",
    "fuel": "Travel",

    "hospital": "Health",
    "clinic": "Health",
    "medicine": "Health"

}


def predict(note):

    if not isinstance(note, str):
        return "Unknown"

    note = note.lower()

    for keyword, category in rules.items():

        if keyword in note:

            return category

    return "Unknown"
