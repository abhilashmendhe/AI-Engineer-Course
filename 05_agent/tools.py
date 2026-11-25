def get_weather(city: str) -> dict:
    """get the current weather for a city
    Arg:
        city: the name of the city
    """
    print("Called weather tool")
    return {"city": city, "weather": "sunny 34C"}
