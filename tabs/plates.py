import streamlit as st
from streamlit_tags import st_tags
from base.key import encrypt_and_save, decrypt_and_load
import os
import re

ENC_FILE = "keywords.enc"

# Format kontrolü: Sadece harf ve rakam içeren tagler kabul edilsin
def is_valid_keyword(keyword):
    # Plaka formatı: 2 rakam + 1-3 harf + 2-5 rakam (Örnek: 34ABC1234)
    
    return bool(re.match(r'^[0-9]{2}[A-Z]{1,3}[0-9]{2,5}$', keyword.replace(" ", "").upper()))

# Doğrulama fonksiyonu
def validate_keywords(keywords):
    valid_keywords = [kw.replace(" ", "").upper() for kw in keywords if is_valid_keyword(kw)]
    invalid_keywords = [kw.replace(" ", "").upper() for kw in keywords if not is_valid_keyword(kw)]
    return valid_keywords, invalid_keywords

# Dosya mevcutsa şifrelenmiş veriyi çöz ve yükle, yoksa boş liste oluştur
if os.path.exists(ENC_FILE):
    keys = decrypt_and_load(ENC_FILE)
else:
    keys = []
    encrypt_and_save(keys, ENC_FILE)

# Kullanıcıdan keyword inputlarını al
keywords = st_tags(
    label='# Enter Keywords:',
    text='Press enter to add more',
    value=keys,
    suggestions=['five', 'six', 'seven',
                 'eight', 'nine', 'three',
                 'eleven', 'ten', 'four'],
    maxtags=40,
    key='1'
)

# Doğrulama işlemi
valid_keywords, invalid_keywords = validate_keywords(keywords)

# Geçersiz tagler varsa kullanıcıya uyarı göster
if invalid_keywords:
    st.error(f"Geçersiz tagler: {', '.join(invalid_keywords)}. Lütfen yalnızca plakayı girin (Örneğin : 01ABC234).")

# Sadece geçerli tagleri kaydet
if valid_keywords != keys:  # Eğer yeni bir şey eklenirse
    encrypt_and_save(valid_keywords, ENC_FILE)




