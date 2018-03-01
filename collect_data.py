#!/usr/bin/env python

import math
import random
import numpy as np
from pysam import AlignmentFile

def pint(str, start):
    for i in range(start, len(str)):
        if not '0' <= str[i] <= '9':
            print(str[i])
            break
    return i

def most_significant_mut_count(seq):
    i, counter = 0, [0] * 6

    while i < len(seq):
        c = seq[i].upper()

        try:
            m = "ATCG+-".index(c)

            if m >= 4:
                counter[m] += 1
                j = pint(seq, i+1)
                i = j + int(seq[i+1:j])
            else:
                counter[m] += 1
                i += 1
        except ValueError:
            if c == '^':
                i += 2
            else:
                i += 1

    return max(counter)

def read_gdna(pileup):
    with open(pileup) as f:
        for line in f:
            line = line.split()
            
            depth = int(line[3])
            if depth > 80:
                freq = most_significant_mut_count(line[4]) / depth

                if freq < 0.01:
                    yield line[0], int(line[1]), 0
                elif freq > 0.24 or (depth > 200 and freq > 0.2):
                    yield line[0], int(line[1]), 1

def encode_base(b):
    return get({'A': 1, 'T': 2, 'C': 3, 'G': 4}, b, 0)

def decode_base(b):
    return "NATCG"[b]

def encode_read(read, center):
    image = np.zeros((64, 10), float)

    def fill_pixel(f, pos, *args):
        if center - 31 <= pos <= center + 31:
            f(pos - center + 31, *args)

    def pixel_match(offset, ref):
        alt = encode_base(ref) - 1
        if alt != -1
            image[offset, alt] = min(read.qual[relpos], 60) / 60
            image[offset, alt + 6] = 1

    def pixel_insert(offset, len):
        image[offset, 4] = min(len, 20) / 20

    def pixel_delete(offset, ref):
        image[offset, 5] = 1
        image[offset, encode_base(ref) + 5] = 1

    insertion = 0
    for relpos, refpos, refbase in read.get_aligned_pairs(with_seq=True):
        if refpos == None:
            insertion += 1
        elif relpos == None:
            fill_pixel(pixel_delete, refpos)
        else:
            fill_pixel(pixel_match, refpos, refbase.upper())

            if insertion > 0:
                fill_pixel(pixel_insert, refpos)
                insertion = 0

    tlen = read.template_length

    image[63, 0] = min(read.mapping_quality, 60) / 60
    image[63, 1] = read.is_reverse
    image[63, 2] = read.is_read1
    image[63, 3] = tlen / 640 if tlen < 256 else math.log(min(tlen, 1048576), 2) / 20
    image[63, 4] = read.is_secondary or read.is_supplementary

    return image

def collect_data(bam, pileup, image, txt):
    out_image, out_txt = open(image, 'w'), open(txt, 'w')

    for chr, pos, genotype in read_gdna(pileup):
        reads = list(AlignmentFile(bam).fetch(chr, pos, pos+1))
        depth = len(reads)
        if depth < 64:
            continue

        images = [encode_read(read, pos) for read in reads]

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

        maxsupport = max(counter)
        if maxsupport == 0:
            continue

        mut, freq = "ATCG+-"[counter.index(maxsupport)], maxsupport / depth
        if freq < 0.02 or freq > 0.48: # drop easy samples to save training time
            continue
        elif (freq < 0.06 and genotype == 0 and random.random() < 0.995):
            continue
        elif (freq > 0.24 and genotype == 1 and random.random() < 0.8):
            continue
        
        for image in images:
            image.tofile(out_image)
            print(chr, pos, mut, genotype, freq, depth, sep='\t', file=out_txt)

if __name__ == '__main__':
    import fire
    fire.Fire(collect_data)