import hashlib
from itertools import islice

def signup(email, pwd, name):
    enc = pwd.encode()
    hash1 = hashlib.md5(enc).hexdigest()

    
    with open("credentials.txt", "a+") as f:
        for line in f:
            if line == email:
                return False, "woops"
        f.write(email + "\n")
        f.write(hash1 + "\n")
        f.write(name + "\n")
        f.close()

        return True, name


def login(email, pwd):
    auth = pwd.encode()
    auth_hash = hashlib.md5(auth).hexdigest()
    with open("credentials.txt", "r") as f:
        lines_gen = islice(f, 3)
        count = 0
        for line in lines_gen:
            if count == 0:
                stored_email = line.strip()
            if count == 1:
                stored_pwd = line.strip()
            if count == 2:
                stored_name = line.strip()
            count += 1
        f.close()


    if email == stored_email and auth_hash == stored_pwd:
        return True, stored_name
    else:
        return False, "woops"

