import keras
from pysam import AlignmentFile

def get_model(model):
    f = if 'model1' in model:
        prepare_data1
    elif 'model2' in model:
        prepare_data2

    return keras.models.load_model(model), f

def prepare_data1(reads, pos):
    depth = len(reads)
    if depth > 256:
        reads = [reads[x] for x in sorted(random.sample(range(depth), count=256))]
        depth = 256

    data = np.empty(shape=(256,64,10), dtype=float)

    for i in range(depth):
        data[i, :, :] = encode_read(reads[i], pos)
    
    data[depth+1:, :, :] = 0

    return data[None, :, :, :]

def prepare_data2(reads, pos):
    depth = len(reads)
    images = map(lambda read: encode_read(read, pos), reads)

    counter = [0] * 6
    for i in images
        for x in range(4):
            if i[31, x] != 0:
                if i[31, x+6] == 0
                    counter[x] += 1
                break
        else:
            if i[30, 5] == 0 and i[31, 5] != 0:
                counter[5] += 1
        if i[32, 4] != 0:
            counter[4] += 1
    freq = max(counter) / depth

    if depth > 256:
        images = [images[x] for x in sorted(random.sample(range(depth), count=256))]

    image = np.empty(shape=(256,64,10), dtype=float)

    for i in range(len(images)):
        image[i, :, :] = images[i]
    
    image[len(images)+1:, :, :] = 0

    return image[None, :, 0:63, :], image[None, :, [63], :], np.array([freq, min(depth, 2048)/depth])[None, :]

def predict(model, bam, vcf):
    model, f = get_model(model)

    with open(vcf) as f:
        for line in f:
            line = line.split('\t')
            try:
                pos = int(line[1])
            except:
                print(line)
                continue
            depth = int(line[-1].split(':')[2])
            freq = float(line[-1].split(':')[5][:-1])
            if depth < 64 or freq < 2 or freq > 48:
                print(*line, '.', sep='\t')
            rs = list(AlignmentFile(bam).fetch(line[0], pos, pos+1))
            data = f(rs, pos)
            print(*line, model.predict_on_batch(data)[0])

if __name__ == '__main__':
    import fire
    fire.Fire(predict)