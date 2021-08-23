from google.cloud import translate

# Instantiates a client
translate_client = translate.Client()

# The text to translate
# text = u'Hello, world!'
text = u'tengo hambre'
# The target language
# target = 'ru'
target = 'ko'

# Translates some text into target
translation = translate_client.translate(text, target_language=target)

print(u'Text: {}'.format(text))
print(u'Translation: {}'.format(translation['translatedText']))
# [END translate_quickstart]