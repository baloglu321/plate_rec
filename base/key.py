from cryptography.fernet import Fernet
import pickle
import os
# Anahtar oluşturma ve kaydetme
def create_key():
    key = Fernet.generate_key()
    with open("secret.key", "wb") as key_file:
        key_file.write(key)

# Anahtar yükleme
def load_key():
    if os.path.exists("secret.key"):
        with open("secret.key", "rb") as key_file:
            return key_file.read()
    else:
        create_key()
        with open("secret.key", "rb") as key_file:
            return key_file.read()
        
# Veriyi şifreleme ve kaydetme
def encrypt_and_save(data, filename):
    key = load_key()
    fernet = Fernet(key)
    
    # Veriyi binary formatında şifrele
    encrypted_data = fernet.encrypt(pickle.dumps(data))
    
    # Şifreli veriyi dosyaya yaz
    with open(filename, "wb") as file:
        file.write(encrypted_data)

# Şifreli veriyi okuma ve çözme
def decrypt_and_load(filename):
    key = load_key()
    fernet = Fernet(key)
    
    # Şifreli veriyi dosyadan oku
    with open(filename, "rb") as file:
        encrypted_data = file.read()
    
    # Veriyi çöz
    return pickle.loads(fernet.decrypt(encrypted_data))





