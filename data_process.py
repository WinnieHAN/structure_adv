import os, sys, torch, codecs
# from trainer import *

def main(maintype):
    # 'pascalprocess'
    if maintype=='fast2conll':
        def dep_from_hdpdep_output(dep_in):
            deps_p_c = dep_in[1:]
            deps = {}
            for pc in deps_p_c:
                [p, c] = [int(i) for i in pc.split('-')]
                if p == len(deps_p_c):
                    deps[c] = 0
                else:
                    deps[c] = p + 1
            return deps

        def search(path, word):
            for filename in os.listdir(path):
                fp = os.path.join(path, filename)
                if word in filename:
                    return fp
                elif os.path.isdir(fp):
                    search(fp, word)


        langs = ['arabic', 'basque', 'czech', 'danish', 'dutch', 'portuguese', 'slovene', 'swedish']
        mainPath = '/home/hanwj/discrimitive_dmv_1/data/pascal-pos'
        instc = 'train_init_conll'
        intdep = '/home/hanwj/hdp_dep/code_'
        outstc = 'train_hdpdep_init_conll'

        for lang in langs:
            in_stc = os.path.join(os.path.join(mainPath, lang), instc)
            runfile = search(intdep + lang, 'run__')
            in_dep = runfile+'/out_put'
            out_stc = os.path.join(os.path.join(mainPath, lang), outstc)
            stcs = codecs.open(in_stc, 'r', encoding='utf8').read().rstrip('\n').split('\n\n')
            token_stcs = [[[k for k in line.rstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs]
            stcs_num = len(token_stcs)
            print(lang + '  num:  ' + str(stcs_num))

            stcs_dep = codecs.open(in_dep, 'r', encoding='utf8').read().rstrip('\n').split('\n\n')
            token_stcs_dep = [[[k for k in line.rstrip(' ').split(' ')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs_dep]

            wf = codecs.open(out_stc, 'w', encoding='utf8')
            for i in range(stcs_num):
                token_dep = dep_from_hdpdep_output(token_stcs_dep[i][3]) # read
                for j in range(len(token_stcs[i])):  # one sentences
                    line = token_stcs[i][j][0] + '\t' + token_stcs[i][j][1] + '\t' + token_stcs[i][j][2] + '\t' + token_stcs[i][j][3] + '\t' + \
                           token_stcs[i][j][4] + '\t' + token_stcs[i][j][5] + '\t' + str(token_dep[j]) + '\n'
                    wf.write(line)
                wf.write('\n')
            wf.close()


    elif maintype=='findtags':
        tagdic = {}
        num_tag = 0
        path = '/home/hanwj/discrimitive_dmv_1/data/pascal-pos-forHDPDEP/'
        langs = ['arabic', 'basque', 'czech', 'danish', 'dutch', 'portuguese', 'slovene', 'swedish']

        for lang in langs:
            stcs = codecs.open(path+lang+'/poses_english', 'r', encoding='utf8').read().rstrip('\n').split('\n')
            token_stcs = [[j for j in i.rstrip(' ').split(' ')] for i in stcs]
            for i in token_stcs:
                for j in i:
                    if j not in tagdic:
                        tagdic[j] = num_tag
                        num_tag = num_tag + 1
        print(tagdic)

    elif maintype=='conllforHDPDEP':
        mainPath = '/home/hanwj/discrimitive_dmv_1/data/pascal-pos-forHDPDEP'
        langs = ['arabic', 'basque', 'czech', 'danish', 'dutch', 'portuguese', 'slovene', 'swedish']
        # ['test', 'test-all']  '_upos_conll'
        instc = 'train_upos_conll'  # 'test-all_upos_conll'
        outtags = 'poses_english'
        outwords = 'words_english'
        outdeps = 'deps_english'
        for lang in langs:
            in_stc = os.path.join(os.path.join(mainPath, lang), instc)
            stcs = codecs.open(in_stc, 'r', encoding='utf8').read().rstrip('\n').split('\n\n')
            token_stcs = [[[k for k in line.rstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs]
            stcs_num = len(token_stcs)
            print(lang + '  num:  ' + str(stcs_num))

            out_tags = os.path.join(os.path.join(mainPath, lang), outtags)
            out_words = os.path.join(os.path.join(mainPath, lang), outwords)
            out_deps = os.path.join(os.path.join(mainPath, lang), outdeps)

            wf = codecs.open(out_tags, 'w', encoding='utf8')
            for i in range(stcs_num):
                for j in range(len(token_stcs[i])):  # one sentences
                    line = token_stcs[i][j][4] + ' '
                    wf.write(line)
                wf.write('#\n')
            wf.close()

            wf = codecs.open(out_words, 'w', encoding='utf8')
            for i in range(stcs_num):
                for j in range(len(token_stcs[i])):  # one sentences
                    line = token_stcs[i][j][1] + ' '
                    wf.write(line)
                wf.write('#\n')
            wf.close()

            wf = codecs.open(out_deps, 'w', encoding='utf8')
            for i in range(stcs_num):
                for j in range(len(token_stcs[i])):  # one sentences
                    par = 0
                    if j==0:
                        j = len(token_stcs[i])
                    # if int(token_stcs[i][j][6])==0:
                    #     par = len(token_stcs[i])
                    # else:
                    #     par = int(token_stcs[i][j][6]) - 1
                    line = str(par) + '-' + str(j) + ' '
                    wf.write(line)
                wf.write('\n')
            wf.close()


    elif maintype=='pascalinit_utf':
        mainPath = '/home/hanwj/discrimitive_dmv_1/data/pascal-pos'
        langs = ['arabic', 'basque', 'czech', 'danish', 'dutch', 'portuguese', 'slovene', 'swedish']
        instc = 'train_init_conll'
        intdep = 'train_ndmv_init_conll'
        outstc = 'train_ndmv_init_conll_new'
        for lang in langs:
            in_stc = os.path.join(os.path.join(mainPath, lang), instc)
            in_dep = os.path.join(os.path.join(mainPath, lang), intdep)
            out_stc = os.path.join(os.path.join(mainPath, lang), outstc)
            stcs = codecs.open(in_stc, 'r', encoding='utf8').read().rstrip('\n').split('\n\n')
            token_stcs = [[[k for k in line.rstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs]
            stcs_num = len(token_stcs)
            print(lang + '  num:  ' + str(stcs_num))
            stcs_dep = codecs.open(in_dep, 'r', encoding='utf8').read().rstrip('\n').split('\n\n')
            token_stcs_dep = [[[k for k in line.rstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs_dep]

            wf = codecs.open(out_stc, 'w', encoding='utf8')
            for i in range(stcs_num):
                for j in range(len(token_stcs[i])):  # one sentences
                    line = token_stcs[i][j][0] + '\t' + token_stcs[i][j][1] + '\t' + token_stcs[i][j][2] + '\t' + token_stcs[i][j][3] + '\t' + \
                           token_stcs[i][j][4] + '\t' + token_stcs[i][j][5] + \
                           '\t' + token_stcs_dep[i][j][6] + '\n'
                    wf.write(line)
                wf.write('\n')
            wf.close()


    elif maintype=='pascalprocess':
        mainPath = '/home/hanwj/discrimitive_dmv_1/data/pascal-pos-forHDPDEP'
        langs = ['arabic', 'basque', 'english', 'childes', 'czech', 'danish', 'dutch', 'portuguese', 'slovene', 'swedish']

        for train_dev_test in ['train']:  # ['train', 'dev', 'test', 'test-all']:
            for lang in langs:
                f_name = os.path.join(os.path.join(mainPath, lang), train_dev_test)
                f_name_w = os.path.join(os.path.join(mainPath, lang), train_dev_test + '_upos_conll')
                stcs = open(f_name).read().rstrip("\n").split("\n\n")

                # temp1_sts = [stc.rstrip("\n").rsplit("\n") for stc in stcs]
                # temp2_sts = [line.rstrip('\t').split('\t') for gg]

                token_stcs = [[[k for k in line.rstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs]

                stcs_num = len(token_stcs)
                print(lang + '  num:  ' + str(stcs_num))

                position_word = 0
                position_ftag = 1
                position_utag = 2
                position_parent = 3

                wf = open(f_name_w, 'w')

                for i in range(stcs_num):
                    for j in range(len(token_stcs[i][0])):  # one sentences
                        line = '_'+'\t' + token_stcs[i][position_word][j] + '\t' + '_' + '\t' + '_' + '\t' + token_stcs[i][position_utag][j] + '\t' + '_' + \
                               '\t' +token_stcs[i][position_parent][j] + '\n'
                        wf.write(line)
                    wf.write('\n')

                wf.close()
                # abstractSts2File_dev = "data/forWord2Vec/wsj-inf_2-21_dep_filter_10_abstractSts_dev"
                # sentences_dev = open(os.path.join(nowPath, abstractSts2File_dev)).read().rstrip("\n").split("\n")
                # sentences_words_dev = [[int(j) for j in i.rstrip("\t").split('\t')] for i in sentences_dev if
                #                        len(i.rstrip("\t").split('\t')) < 11]
                # sentence_lens_dev = [len(j) for j in sentences_words_dev]
    elif maintype == 'loadmodel_sentencerepresent':
        accIdx = str(162)
        decisionValency = 2
        nowPath = os.getcwd()

        PATH_chd = os.path.join(nowPath, 'wsj_graph/my_model_chd_' + accIdx + '_iter_4.h5')
        PATH_dec = os.path.join(nowPath, 'wsj_graph/my_model_dec_' + accIdx + '_iter_4.h5')
        model_chd = torch.load(PATH_chd)
        model_dec = torch.load(PATH_dec)

        # train_chd = os.path.join(nowPath, 'temp/predicted_train_chd' + accIdx + '.txt')
        # train_dec = os.path.join(nowPath, 'temp/predicted_train_dec' + accIdx + '.txt')

        # abstractSts2File = "wsj_graph/train_parsing162.txt"  # should be corrected !!
        # sentences_train = open(os.path.join(nowPath, abstractSts2File)).read().rstrip("\n").split("\n")
        # sentences_words_train = [[int(j) for j in i.rstrip("\t").split('\t')] for i in sentences_train]
        # sentence_lens_train = [len(j) for j in sentences_words_train]

        dic = open(os.path.join(nowPath, 'wsj_graph/dic162.txt')).read().rstrip("\n").split("\n")
        dic2pairlist = [[j for j in i.rstrip("\t").split("\t")] for i in dic]
        dic2pair = {}
        for i in dic2pairlist:
            dic2pair[int(i[2])] = i[0]
        print(dic2pair)

        abstractSts2File = "wsj_graph/train_parsing162.txt"  # should be corrected !!
        sentences_train = open(os.path.join(nowPath, abstractSts2File)).read().rstrip("\n").split("\n\n")
        sentences_words_train = [[j.rstrip("\t").split('\t')[1] for j in i.rstrip("\n").split('\n')] for i in sentences_train]
        sentences_tags_idx_train = [[int(j.rstrip("\t").split('\t')[4]) for j in i.rstrip("\n").split('\n')] for i in sentences_train]
        sentences_tags_train = [[dic2pair[int(j.rstrip("\t").split('\t')[4])] for j in i.rstrip("\n").split('\n')] for i in sentences_train]

        sentence_lens_train = [len(j) for j in sentences_words_train]

        model_chd.eval()
        model_dec.eval()

        predictSentence(NN_child=model_chd, NN_decision=model_dec,
                          # predicted_chd=train_chd,
                          # predicted_dec=train_dec,
                          sentences_words=sentences_tags_idx_train,
                          sentence_lens=sentence_lens_train,
                          sentences_posSeq=sentences_tags_idx_train,
                          valency_size=decisionValency)

        wf_stc_w = open(os.path.join(nowPath, "wsj_graph/stc_words.txt"), 'w')
        line_temp = ['_'.join(i) for i in sentences_words_train]
        line = '\n'.join(line_temp)
        wf_stc_w.write(line)
        wf_stc_w.close()

        wf_stc_w = open(os.path.join(nowPath, "wsj_graph/stc_postags.txt"), 'w')
        line_temp = ['_'.join(i) for i in sentences_tags_train]
        line = '\n'.join(line_temp)
        wf_stc_w.write(line)
        wf_stc_w.close()

        wf_stc_w = open(os.path.join(nowPath, "wsj_graph/stc_words_postags.txt"), 'w')
        line_temp_t = ['_'.join(i) for i in sentences_tags_train]
        line_temp_w = ['_'.join(i) for i in sentences_words_train]
        line_temp = []
        for i in range(len(sentences_words_train)):
            line_temp.append(line_temp_t[i]+'/'+line_temp_w[i])
        line = '\n'.join(line_temp)
        wf_stc_w.write(line)
        wf_stc_w.close()


    elif maintype == 'loadmodel_rule_represent':
        accIdx = str(162)
        decisionValency = 2
        nowPath = os.getcwd()

        PATH_chd = os.path.join(nowPath, 'wsj_graph/my_model_chd_' + accIdx + '_iter_4.h5')
        PATH_dec = os.path.join(nowPath, 'wsj_graph/my_model_dec_' + accIdx + '_iter_4.h5')
        model_chd = torch.load(PATH_chd)
        model_dec = torch.load(PATH_dec)

        dic = open(os.path.join(nowPath, 'wsj_graph/dic162.txt')).read().rstrip("\n").split("\n")
        dic2pairlist = [[j for j in i.rstrip("\t").split("\t")] for i in dic]
        dic2pair = {}
        for i in dic2pairlist:
            dic2pair[int(i[2])] = i[0]
        print(dic2pair)

        # stcs = open(file, 'rU').read().rstrip('\n').split('\n\n')
        # token_stcs = [[[k[1].lower() for k in line.rstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs]

        # for i in range(len(token_stcs)):
        #     pass

        abstractSts2File = "wsj_graph/train_parsing162.txt"  # should be corrected !!
        sentences_train = open(os.path.join(nowPath, abstractSts2File)).read().rstrip("\n").split("\n\n")
        sentences_words_train = [[j.rstrip("\t").split('\t')[1] for j in i.rstrip("\n").split('\n')] for i in sentences_train]
        sentences_tags_idx_train = [[int(j.rstrip("\t").split('\t')[4]) for j in i.rstrip("\n").split('\n')] for i in sentences_train]
        sentences_tags_train = [[dic2pair[int(j.rstrip("\t").split('\t')[4])] for j in i.rstrip("\n").split('\n')] for i in sentences_train]

        sentence_lens_train = [len(j) for j in sentences_words_train]

        toy_tags_idx = [sentences_tags_idx_train[12], sentences_tags_idx_train[15]]
        toy_lens = [len(sentences_tags_idx_train[12]), len(sentences_tags_idx_train[15])]

        model_chd.eval()
        model_dec.eval()

        probs = predictRules(NN_child=model_chd, NN_decision=model_dec,
                          sentences_words=sentences_tags_idx_train,
                          sentence_lens=sentence_lens_train,
                          sentences_posSeq=sentences_tags_idx_train,
                          valency_size=decisionValency)

        print('probs:  ')
        print(probs)
    elif main_type=='ud_len_limit':
        import numpy as np
        langsPathes = np.array(['UD_Ancient_Greek', 'UD_Ancient_Greek-PROIEL', 'UD_Arabic', 'UD_Basque', 'UD_Bulgarian',
                                'UD_Catalan', 'UD_Chinese', 'UD_Coptic', 'UD_Croatian', 'UD_Czech', 'UD_Czech-CAC',
                                'UD_Czech-CLTT', 'UD_Danish', 'UD_Dutch', 'UD_Dutch-LassySmall', 'UD_English',
                                'UD_English-ESL', 'UD_English-LinES', 'UD_Estonian', 'UD_Finnish', 'UD_Finnish-FTB',
                                'UD_French', 'UD_Galician', 'UD_Galician-TreeGal', 'UD_German', 'UD_Gothic', 'UD_Greek',
                                'UD_Hebrew', 'UD_Hindi', 'UD_Hungarian', 'UD_Indonesian', 'UD_Irish', 'UD_Italian',
                                'UD_Japanese', 'UD_Japanese-KTC', 'UD_Kazakh', 'UD_Latin', 'UD_Latin-ITTB',
                                'UD_Latin-PROIEL',
                                'UD_Latvian', 'UD_Norwegian', 'UD_Old_Church_Slavonic', 'UD_Persian', 'UD_Polish',
                                'UD_Portuguese',
                                'UD_Portuguese-Bosque', 'UD_Portuguese-BR', 'UD_Romanian', 'UD_Russian',
                                'UD_Russian-SynTagRus',
                                'UD_Sanskrit', 'UD_Slovak', 'UD_Slovenian', 'UD_Slovenian-SST', 'UD_Spanish',
                                'UD_Spanish-AnCora',
                                'UD_Swedish', 'UD_Swedish-LinES', 'UD_Swedish_Sign_Language', 'UD_Tamil', 'UD_Turkish',
                                'UD_Ukrainian', 'UD_Uyghur', 'UD_Vietnamese'])
        i2langs = np.array(
            ['grc', 'grc_proiel', 'ar', 'eu', 'bg', 'ca', 'zh', 'cop', 'hr', 'cs', 'cs_cac', 'cs_cltt', 'da',
             'nl', 'nl_lassysmall', 'en', 'en_esl', 'en_lines', 'et', 'fi', 'fi_ftb', 'fr', 'gl', 'gl_treegal',
             'de', 'got', 'el', 'he', 'hi', 'hu', 'id', 'ga', 'it', 'ja', 'ja_ktc', 'kk', 'la', 'la_ittb',
             'la_proiel',
             'lv', 'no', 'cu', 'fa', 'pl', 'pt', 'pt_bosque', 'pt_br', 'ro', 'ru', 'ru_syntagrus', 'sa', 'sk',
             'sl',
             'sl_sst', 'es', 'es_ancora', 'sv', 'sv_lines', 'swl', 'ta', 'tr', 'uk', 'ug', 'vi'])

        langs2i = {i: j for j, i in enumerate(i2langs)}

        mainPath = '/home/hanwj/Code/ddmv/data/ud-treebanks-v1.4'
        # instc = 'train_init_conll'
        # intdep = 'train_ndmv_init_conll'
        # outstc = 'train_ndmv_init_conll_new'
        for lang in range(len(langs2i)):#i2langs:
            in_stc = os.path.join(os.path.join(mainPath, langsPathes[lang]), i2langs[lang]+'-ud-test-nopunct-len40.conllu')

            out_stc = os.path.join(os.path.join(mainPath, langsPathes[lang]), i2langs[lang]+'-ud-test-nopunct-len10.conllu')
            stcs = codecs.open(in_stc, 'r', encoding='utf8').read().rstrip('\n').lstrip('\n').split('\n\n')
            stcs = [i.rstrip('\n').lstrip('\n') for i in stcs]
            token_stcs = [[[k for k in line.rstrip('\t').lstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs if len(stc.rstrip("\n").rsplit("\n"))< 11]
            stcs_num = len(token_stcs)
            print(str(lang)+':  '+ langsPathes[lang] + '  num:  ' + str(stcs_num))
            # stcs_dep = codecs.open(in_dep, 'r', encoding='utf8').read().rstrip('\n').split('\n\n')
            # token_stcs_dep = [[[k for k in line.rstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs_dep]

            wf = codecs.open(out_stc, 'w', encoding='utf8')
            for i in range(stcs_num):
                # print(i)
                if not token_stcs[i] == [['']]:
                    for j in range(len(token_stcs[i])):  # one sentences
                        line = token_stcs[i][j][0] + '\t' + token_stcs[i][j][1] + '\t' + token_stcs[i][j][2] + '\t' + token_stcs[i][j][3] + '\t' + \
                               token_stcs[i][j][4] + '\t' + token_stcs[i][j][5] + \
                               '\t' + token_stcs[i][j][6] + '\t' + token_stcs[i][j][7] + '\t' + token_stcs[i][j][8] + '\t' + token_stcs[i][j][9] + '\n'
                        wf.write(line)
                    wf.write('\n')
            wf.close()

    elif maintype=='conllu_seq2seq':
        instc = 'data/ptb/dev.conllu'
        outstc = 'data/ptb/dev_seq2seq.conllu'

        stcs = codecs.open(instc, 'r', encoding='utf8').read().rstrip('\n').split('\n\n')
        token_stcs = [[[k for k in line.rstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs]
        stcs_num = len(token_stcs)

        wf = codecs.open(outstc, 'w', encoding='utf8')
        for i in range(stcs_num):
            line = []
            for j in range(len(token_stcs[i])):  # one sentences
                line.append(token_stcs[i][j][1])
            new_line = ' '.join(line) + '\t' + ' '.join(line)
            wf.write(new_line)
            wf.write('\n')
        wf.close()

    elif main_type=='tab_to_space':
        instc = 'data/ctb/test.conllu'
        outstc = 'data/ctb/test.conllu'

        stcs = codecs.open(instc, 'r', encoding='utf8').read().rstrip('\n').split('\n\n')
        token_stcs = [[[k for k in line.rstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs]
        stcs_num = len(token_stcs)
        res = []
        # wf = codecs.open(outstc, 'w', encoding='utf8')
        for i in range(stcs_num):
            print(i)
            line = []
            for j in range(len(token_stcs[i])):  # one sentences
                if  token_stcs[i][j][4]=='PU' and (not token_stcs[i][j][1] in res):
                    res.append(token_stcs[i][j][1])

                # wf.write(' '.join(token_stcs[i][j])+'\n')
            # wf.write('\n')
        # wf.close()
        print(res)

if __name__ == "__main__":
    main_type = 'tab_to_space' #'conllu_seq2seq' #'ud_len_limit' #''loadmodel_rule_represent' #'conllforHDPDEP' #'pascalprocess'  # pascalinit_utf  #  loadmodel_sentencerepresent  # 'pascalprocess'
    main(main_type)