###generate negpeptide for train peptide
import random, json, os
from collections import OrderedDict

fatherpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def generateneg(length, trainpeptide):
    currentpath = os.getcwd()
    unusualAA = ['B', 'J', 'O', 'U', 'X', 'Z']
    postrainpeptide = []
    with open('%s/source/IEDBSourceprotein_Tcellassay-ligand.json' % fatherpath, 'r') as f:
        dict_IDfasta = json.load(f)

    for eachpeptide in trainpeptide.keys():
        postrainpeptide.append(eachpeptide)  # 存放所有的peptide

    dict_possetneg = OrderedDict()
    for eachpos in postrainpeptide:
        dict_possetneg.setdefault('pos', []).append(eachpos)
    dict_possetneg['neg'] = OrderedDict()

    numpos = len(postrainpeptide)
    neg_list_refine = []
    while len(neg_list_refine) <= numpos:
        thisfastacorrect = 0
        while thisfastacorrect == 0:
            choiceID = (random.sample(dict_IDfasta.keys(), 1))[0]
            choicefasta = dict_IDfasta[choiceID]
            if len(choicefasta) < int(length):
                continue
            else:
                havetime = 0
                for eachAA in unusualAA:
                    if eachAA in choicefasta:
                        havetime += 1
                if havetime >= 1:
                    continue
            startpos, stoppos = 0, int(length)
            choicelist = []
            while stoppos < len(choicefasta) + 1:
                choicelist.append(choicefasta[startpos: stoppos])
                startpos += 1
                stoppos += 1
            correct = 0
            choicetime = 0
            while correct == 0:
                choiceseg = (random.sample(choicelist, 1))[0]
                if choiceseg not in postrainpeptide and choiceseg not in neg_list_refine:
                    neg_list_refine.append(choiceseg)
                    correct += 1
                    thisfastacorrect += 1
                else:
                    choicetime += 1
                    if choicetime == len(choicelist) * 2:
                        correct += 1
    dict_possetneg['neg'] = neg_list_refine
    return dict_possetneg

