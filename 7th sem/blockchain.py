import hashlib
def sha256_f(data):
    sha256 = hashlib.sha256()
    sha256.update(data.encode('utf-8'))
    return sha256.hexdigest()


def merkletree(data_list):
    hashes = [sha256_f(data) for data in data_list]
    print("Leafs: ", hashes)
    while len(hashes) > 1:
        if len(hashes) % 2 != 0:
            hashes.append(hashes[-1])
        new_hashes = []
        for i in range(0, len(hashes), 2):
            combined_hash = sha256_f(hashes[i] + hashes[i + 1])
            new_hashes.append(combined_hash)
        hashes = new_hashes
        print("Parents: ", hashes)
    return hashes[0]

data_list = ["Gulsezim", "Bota", "Dima", "Malika", "Aidana", "Daniil"]

root_hash = merkletree(data_list)
print("Merkle tree root:", root_hash)
