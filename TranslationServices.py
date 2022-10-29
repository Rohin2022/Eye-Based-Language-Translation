import translators as ts

def translate(toTranslate,target_lang):
    translation = ts.google(toTranslate, from_language='hi', to_language=target_lang)
    return translation