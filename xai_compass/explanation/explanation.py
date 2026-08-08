ENGLISH_TEMPLATES = {
    "Liver Disease": {
        "meaning": "The AetherDx model has classified the uploaded nail image under the Liver Disease category based on visual patterns learned from its training data.",
        "context": "Certain nail changes can be associated with systemic conditions, including liver-related disorders. Examples described in medical literature include changes in nail color, nail-bed appearance, or other alterations in the nail unit. These findings are NOT specific enough on their own to confirm liver disease.",
    },
    "Psoriasis": {
        "meaning": "The AetherDx model has classified the uploaded nail image under the Psoriasis category based on visual patterns learned from its training data.",
        "context": "Psoriasis can affect the nails, causing changes that are commonly described in medical literature, such as pitting, discoloration (oil drop sign), onycholysis (separation of the nail), or subungual hyperkeratosis. These findings are NOT specific enough on their own to confirm a psoriasis diagnosis.",
    },
    "Healthy": {
        "meaning": "The model classified the uploaded nail image as belonging to the Healthy category based on the visual patterns represented in its training data.",
        "context": "A healthy nail appearance typically lacks significant discoloration, structural deformities, or abnormal patterns. However, a normal-appearing nail does not guarantee the absence of systemic or localized conditions.",
    }
}

HINDI_TEMPLATES = {
    "Liver Disease": {
        "meaning": "AetherDx मॉडल ने अपने प्रशिक्षण डेटा से सीखे गए दृश्य पैटर्न के आधार पर अपलोड की गई नाखून छवि को लिवर रोग (Liver Disease) श्रेणी में वर्गीकृत किया है।",
        "context": "नाखूनों में कुछ बदलाव प्रणालीगत स्थितियों से जुड़े हो सकते हैं, जिनमें लिवर से संबंधित विकार शामिल हैं। चिकित्सा साहित्य में वर्णित उदाहरणों में नाखून के रंग, नाखून के बिस्तर (nail-bed) की उपस्थिति, या नाखून इकाई में अन्य परिवर्तन शामिल हैं। ये निष्कर्ष अपने आप में लिवर रोग की पुष्टि करने के लिए पर्याप्त विशिष्ट नहीं हैं।",
    },
    "Psoriasis": {
        "meaning": "AetherDx मॉडल ने अपने प्रशिक्षण डेटा से सीखे गए दृश्य पैटर्न के आधार पर अपलोड की गई नाखून छवि को सोरायसिस (Psoriasis) श्रेणी में वर्गीकृत किया है।",
        "context": "सोरायसिस नाखूनों को प्रभावित कर सकता है, जिससे ऐसे परिवर्तन हो सकते हैं जिन्हें आमतौर पर चिकित्सा साहित्य में वर्णित किया जाता है, जैसे कि गड्ढे पड़ना (pitting), मलिनकिरण, नाखून का अलग होना (onycholysis), या अत्यधिक मोटा होना। ये निष्कर्ष अपने आप में सोरायसिस निदान की पुष्टि करने के लिए पर्याप्त विशिष्ट नहीं हैं।",
    },
    "Healthy": {
        "meaning": "मॉडल ने अपने प्रशिक्षण डेटा में दर्शाए गए दृश्य पैटर्न के आधार पर अपलोड की गई नाखून छवि को स्वस्थ (Healthy) श्रेणी से संबंधित वर्गीकृत किया है।",
        "context": "स्वस्थ नाखून की उपस्थिति में आमतौर पर महत्वपूर्ण मलिनकिरण, संरचनात्मक विकृति या असामान्य पैटर्न का अभाव होता है। हालांकि, एक सामान्य दिखने वाला नाखून प्रणालीगत या स्थानीयकृत स्थितियों की अनुपस्थिति की गारंटी नहीं देता है।",
    }
}

MARATHI_TEMPLATES = {
    "Liver Disease": {
        "meaning": "AetherDx मॉडेलने आपल्या प्रशिक्षण डेटावरून शिकलेल्या दृश्य नमुन्यांवर आधारित अपलोड केलेल्या नखाच्या प्रतिमेचे यकृत रोग (Liver Disease) श्रेणीमध्ये वर्गीकरण केले आहे.",
        "context": "नखांमधील काही बदल यकृताशी संबंधित विकारांसह प्रणालीगत स्थितींशी संबंधित असू शकतात. वैद्यकीय साहित्यात वर्णन केलेल्या उदाहरणांमध्ये नखाचा रंग, नखाच्या पृष्ठभागाचे स्वरूप किंवा नखामध्ये इतर बदल समाविष्ट आहेत. हे निष्कर्ष स्वतःहून यकृताच्या रोगाची पुष्टी करण्यासाठी पुरेसे विशिष्ट नाहीत.",
    },
    "Psoriasis": {
        "meaning": "AetherDx मॉडेलने आपल्या प्रशिक्षण डेटावरून शिकलेल्या दृश्य नमुन्यांवर आधारित अपलोड केलेल्या नखाच्या प्रतिमेचे सोरायसिस (Psoriasis) श्रेणीमध्ये वर्गीकरण केले आहे.",
        "context": "सोरायसिसचा नखांवर परिणाम होऊ शकतो, ज्यामुळे वैद्यकीय साहित्यात वर्णन केलेले बदल होऊ शकतात, जसे की खड्डे पडणे (pitting), रंग बदलणे, नख वेगळे होणे (onycholysis) किंवा जाडी वाढणे. हे निष्कर्ष स्वतःहून सोरायसिस निदानाची पुष्टी करण्यासाठी पुरेसे विशिष्ट नाहीत.",
    },
    "Healthy": {
        "meaning": "मॉडेलने आपल्या प्रशिक्षण डेटावरील दृश्य नमुन्यांवर आधारित अपलोड केलेल्या नखाच्या प्रतिमेचे निरोगी (Healthy) श्रेणीशी संबंधित म्हणून वर्गीकरण केले आहे.",
        "context": "निरोगी नखांच्या उपस्थितीमध्ये सामान्यतः लक्षणीय रंग बदल, संरचनात्मक विकृती किंवा असामान्य नमुन्यांचा अभाव असतो. तथापि, सामान्य दिसणारे नख कोणत्याही आजाराच्या अनुपस्थितीची हमी देत नाही.",
    }
}

def get_explanation(language, prediction, confidence):
    # Map predictions to templates
    pred_key = prediction
    if "Liver" in prediction:
        pred_key = "Liver Disease"
    elif "Psoriasis" in prediction:
        pred_key = "Psoriasis"
    elif "Healthy" in prediction:
        pred_key = "Healthy"
    
    if language == "english":
        tpl = ENGLISH_TEMPLATES.get(pred_key, ENGLISH_TEMPLATES["Healthy"])
        return f"""### What does this prediction mean?
{tpl['meaning']}

### Medical context
{tpl['context']}

### What did the AI focus on?
The Grad-CAM visualization highlights the regions of the nail image that contributed most strongly to the model's classification. The highlighted area represents the model's attention/evidence for the prediction. It does NOT mean that the highlighted region is itself a confirmed medical lesion or that the model has identified a specific disease marker.

### Why this prediction was generated
The model classified the image based on visual patterns learned during training that are associated with the predicted class.

**Model output probability: {confidence}%**
This probability represents the model's output for the selected class. It should NOT be interpreted as a {confidence}% probability that the person clinically has the disease.

### Clinical interpretation
This result is intended for AI-assisted screening/research demonstration only. A qualified healthcare professional should evaluate the patient and consider clinical history, examination, and appropriate diagnostic tests before making a medical decision.
"""
    elif language == "hindi":
        tpl = HINDI_TEMPLATES.get(pred_key, HINDI_TEMPLATES["Healthy"])
        return f"""### इस भविष्यवाणी का क्या अर्थ है?
{tpl['meaning']}

### चिकित्सा संदर्भ
{tpl['context']}

### AI ने किस पर ध्यान केंद्रित किया?
Grad-CAM विज़ुअलाइज़ेशन नाखून की छवि के उन क्षेत्रों को उजागर करता है जिन्होंने मॉडल के वर्गीकरण में सबसे अधिक योगदान दिया। हाइलाइट किया गया क्षेत्र भविष्यवाणी के लिए मॉडल के ध्यान/साक्ष्य का प्रतिनिधित्व करता है। इसका अर्थ यह नहीं है कि हाइलाइट किया गया क्षेत्र अपने आप में एक पुष्ट चिकित्सा घाव है या कि मॉडल ने किसी विशिष्ट रोग मार्कर की पहचान की है।

### यह भविष्यवाणी क्यों उत्पन्न हुई?
मॉडल ने प्रशिक्षण के दौरान सीखे गए दृश्य पैटर्न के आधार पर छवि को वर्गीकृत किया जो अनुमानित वर्ग से जुड़े हैं।

**मॉडल आउटपुट संभावना: {confidence}%**
यह संभावना चयनित वर्ग के लिए मॉडल के आउटपुट का प्रतिनिधित्व करती है। इसे इस रूप में व्याख्या नहीं किया जाना चाहिए कि व्यक्ति को नैदानिक रूप से बीमारी होने की {confidence}% संभावना है।

### नैदानिक व्याख्या
यह परिणाम केवल AI-सहायता प्राप्त स्क्रीनिंग/शोध प्रदर्शन के लिए है। किसी भी चिकित्सा निर्णय लेने से पहले एक योग्य स्वास्थ्य देखभाल पेशेवर को रोगी का मूल्यांकन करना चाहिए और नैदानिक इतिहास, परीक्षा और उपयुक्त नैदानिक परीक्षणों पर विचार करना चाहिए।
"""
    elif language == "marathi":
        tpl = MARATHI_TEMPLATES.get(pred_key, MARATHI_TEMPLATES["Healthy"])
        return f"""### या अंदाजाचा अर्थ काय आहे?
{tpl['meaning']}

### वैद्यकीय संदर्भ
{tpl['context']}

### AI ने कशावर लक्ष केंद्रित केले?
Grad-CAM व्हिज्युअलायझेशन नखाच्या प्रतिमेचे ते भाग हायलाइट करते ज्यांनी मॉडेलच्या वर्गीकरणात सर्वात जास्त योगदान दिले. हायलाइट केलेला भाग अंदाजासाठी मॉडेलचे लक्ष/पुरावा दर्शवतो. याचा अर्थ असा नाही की हायलाइट केलेला भाग हा स्वतःच एक निश्चित वैद्यकीय घाव आहे किंवा मॉडेलने एखादा विशिष्ट रोग मार्कर ओळखला आहे.

### हा अंदाज का निर्माण झाला?
अंदाजित वर्गाशी संबंधित असलेल्या प्रशिक्षणादरम्यान शिकलेल्या दृश्य नमुन्यांवर आधारित मॉडेलने प्रतिमेचे वर्गीकरण केले.

**मॉडेल आउटपुट संभाव्यता: {confidence}%**
ही संभाव्यता निवडलेल्या वर्गासाठी मॉडेलचे आउटपुट दर्शवते. याचा अर्थ असा घेऊ नये की व्यक्तीला क्लिनिकदृष्ट्या तो आजार होण्याची {confidence}% शक्यता आहे.

### वैद्यकीय अर्थ
हा परिणाम केवळ AI-सहाय्यित स्क्रीनिंग/संशोधन प्रात्यक्षिकासाठी आहे. योग्य आरोग्य सेवा व्यावसायिकाने रुग्णाचे मूल्यांकन केले पाहिजे आणि वैद्यकीय निर्णय घेण्यापूर्वी क्लिनिकल इतिहास, तपासणी आणि योग्य निदान चाचण्यांचा विचार केला पाहिजे.
"""
