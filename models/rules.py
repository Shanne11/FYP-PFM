"""
Simple Rule-Based Transaction Categorisation
"""

rules = {

    "electricity": "Bills",
    "water": "Bills",
    "internet": "Bills",

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

    "uber": "Transport",
    "grab": "Transport",
    "taxi": "Transport",
    "bus": "Transport",
    "fuel": "Transport",

    "hospital": "Healthcare",
    "clinic": "Healthcare",
    "medicine": "Healthcare"

}


def predict(note):

    if not isinstance(note, str):
        return "Unknown"

    note = note.lower()

    for keyword, category in rules.items():

        if keyword in note:

            return category

    return "Unknown"