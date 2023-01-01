import hashlib
from itertools import islice


class User:
    def __init__(self, pwd, email, name, resolutions):
        self.pwd = pwd
        self.email = email
        self.name = name
        self.resolutions = resolutions


def signup(email, pwd, name):

    enc = pwd.encode()
    hash1 = hashlib.md5(enc).hexdigest()

    
    with open("credentials.txt", "w+") as f:
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
    stored_email = ""
    stored_pwd = ""
    stored_name=""
    with open("credentials.txt", "r") as f:
        for line in f:
            count = 0
            if line.strip() == email:
                stored_email = line.strip()
                for i in range(2):
                    if count == 0:
                        stored_pwd = next(f).strip()
                        count += 1
                    else:
                        stored_name = next(f).strip()
                        print(stored_name)
                        break
        f.close()

    resolutions = []
    user = User(stored_pwd, stored_email, stored_name, resolutions)
    if email == stored_email and auth_hash == stored_pwd:
        return True, stored_name, user
    else:
        return False, "woops", User("nope", "nope", "nope", resolutions)

def locate_resolutions(user, command, towipe):
    email_to_look = user.email
    resolutions = []
    if command == "add":
        with open("credentials.txt", "r+") as f:
            for line in f:
                count = 0
                if line.strip() == email_to_look:
                    for i in range(2): next(f) # get rid of pwd and name
                    f.write("\n")
                    f.write(towipe)
    if command == "remove":
        with open("credentials.txt", "r") as f:
            lines = f.readlines()
        with open("credentials.txt", "w") as f:
            for line in lines:
                if line.strip("\n") != towipe:
                    f.write(line)

    with open("credentials.txt") as f:
        lines = f.readlines()
        last = lines[-1]

    with open("credentials.txt", "r") as f:
        for line in f:
            if line.strip() == email_to_look:
                for i in range(2): next(f)
                
                for j in range(99999):
                    a = next(f).strip()
                    if '@' not in a and a == last:
                        resolutions.append(a)
                        break
                    elif '@' not in a:
                        resolutions.append(a)
                    else:
                        break
        f.close()

    for i in resolutions:
        print(i)

    return resolutions
