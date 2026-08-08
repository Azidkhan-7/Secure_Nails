def get_explanation(language, prediction, confidence):
    """
    Returns the templated explanation in the given language.
    language should be one of "english", "hindi", "marathi".
    """
    if language == "english":
        return (f"The AetherDx model classified this nail image as {prediction} with {confidence}% confidence. "
                f"The highlighted regions indicate the image areas that contributed most strongly to this prediction.")
    elif language == "hindi":
        return (f"AetherDx मॉडल ने इस नाखून की छवि को {confidence}% विश्वास के साथ {prediction} के रूप में वर्गीकृत किया। "
                f"हाइलाइट किए गए क्षेत्र मॉडल के इस निर्णय में सबसे अधिक योगदान देने वाले हिस्सों को दर्शाते हैं।")
    elif language == "marathi":
        return (f"AetherDx मॉडेलने या नखाच्या प्रतिमेचे {confidence}% विश्वासाने {prediction} म्हणून वर्गीकरण केले. "
                f"हायलाइट केलेले भाग मॉडेलच्या या निर्णयावर सर्वाधिक प्रभाव टाकणारे भाग दर्शवतात.")
    else:
        return ""
