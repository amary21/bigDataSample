import pandas as pd
import time
from math import log, log10


class BayesGizi:

    def __init__(self, dataframe, tipe_gizi):
        self.dataframe = dataframe
        self.tipe_gizi = tipe_gizi

    def filter_gizi(self):
        
        sam_bad = ""
        if self.tipe_gizi == 'Gizi Buruk':
            sam_bad = sam[(sam['diagnosa gizi'] == 'Gizi Buruk')]
        if self.tipe_gizi == 'Gizi Kurang':
            sam_bad = sam[(sam['diagnosa gizi'] == 'Gizi Kurang')]
        if self.tipe_gizi == 'Gizi Normal':
            sam_bad = sam[(sam['diagnosa gizi'] == 'Gizi Normal')]
        if self.tipe_gizi == 'Gizi Lebih':
            sam_bad = sam[(sam['diagnosa gizi'] == 'Gizi Lebih')]

        vars1 = sam_bad.groupby(
            ['umur(bln)', 'BB(kg)', 'gigi', 'penglihatan kanan', 'penglihatan kiri', 'pendengaran kanan', 'pendengaran kiri', 'berbicara']
        ).agg(
            {
                'diagnosa gizi': len,
                'Jenis kelamin': lambda x: x == 'Perempuan'
            }
        ).reset_index()

        return vars1

    def compute(self, genre: str, uage: int, weight: float, gigi: str, side: list=None, snd: list=None, utalk: str='') -> object:
        datasets = self.filter_gizi()
        diag_side = ['cukup', 'baik']
        diag_sound = ['kurang', 'baik']

        # compare genre
        rlgen = ['male', 'female']
        fltgenre = genre if genre in rlgen else ''
        resgenre = False if fltgenre == 'male' else True

        # compare age
        fltage = [x for x in range(60) if x >= int(uage)]
        resage = uage if fltage != [] else 0

        # is float?
        fltweight = True if isinstance(weight, float) else False
        if fltweight is False:
            raise Exception("Opps: Error data type")

        if gigi not in ['cukup', 'baik', 'kurang']:
            gigi = ""
        
        # two list side
        fltside = [x for x in side if x in diag_side]
        rside = fltside if len(fltside) == 2 else ['', '']


        # two list sound
        fltsnd = [x for x in snd if x in diag_sound]
        rsnd = fltsnd if len(fltsnd) == 2 else ['', '']

        if utalk not in ['cukup', 'baik', 'kurang']:
            utalk = ""

        count_genre = datasets[datasets['Jenis kelamin'] == resgenre].sum()
        count_age = datasets[datasets['umur(bln)'] == resage].count()
        by_bb = datasets[datasets['BB(kg)'] == weight].count()
        by_gigi = len(datasets[datasets['gigi'] == gigi]['diagnosa gizi'])
        by_right_side = datasets[datasets['penglihatan kanan'] == rside[0]]
        by_left_side = datasets[datasets['penglihatan kiri'] == rside[1]]
        by_right_sound = datasets[datasets['pendengaran kanan'] == rsnd[0]]
        by_left_sound = datasets[datasets['pendengaran kiri'] == rsnd[1]]
        talk_sound = datasets[datasets['berbicara'] == utalk]

        sum_on = datasets['diagnosa gizi'].count()

        res_genre =count_genre['diagnosa gizi']/sum_on
        res_age =count_age['diagnosa gizi']/sum_on
        res_bb =by_bb['BB(kg)']/sum_on
        res_gigi =by_gigi/sum_on
        res_left_side =by_left_side['diagnosa gizi'].count()/sum_on
        res_right_side =by_right_side['diagnosa gizi'].count()/sum_on
        res_right_sound =by_right_sound['diagnosa gizi'].count()/sum_on
        res_left_sound =by_left_sound['diagnosa gizi'].count()/sum_on
        res_talk =talk_sound['diagnosa gizi'].count()/sum_on

        onlist = [res_genre,res_age, res_bb, res_gigi, res_left_side, res_right_side, res_left_sound, res_right_sound,res_talk]


        to_logs = []
        for dtlist in onlist:
            dtlist = "%.3f".format(dtlist) % dtlist
            if float(dtlist) == 0.0:
                dtlist = 0.001

            to_logs.append(float(dtlist))

        conv_logs = []
        for dtlist in to_logs:
            xlog = log10(dtlist)
            conv_logs.append(xlog)

        return sum(conv_logs)

if __name__ == "__main__":
    sam = pd.read_excel('Status_Gizi_Balita.xlsx')
    print(sam)

    # obj1 = BayesGizi(dataframe=sam, tipe_gizi='Gizi Buruk')
    # obj1.compute('female', uage=44, weight=15.5, gigi='cukup', side=['baik', 'baik'], snd=['baik', 'kurang'], utalk='cukup')

    # obj2 = BayesGizi(dataframe=sam, tipe_gizi='Gizi Kurang')
    # obj2.compute('female', uage=44, weight=15.5, gigi='cukup', side=['baik', 'baik'], snd=['baik', 'kurang'], utalk='cukup')


    # obj3 = BayesGizi(dataframe=sam, tipe_gizi='Gizi Normal')
    # obj3.compute('female', uage=44, weight=15.5, gigi='cukup', side=['baik', 'baik'], snd=['baik', 'kurang'], utalk='cukup')
 
    print("")
    print("=============================================================")

    vjenisKelamin = str(input("Jenis Kelamin [Male\Female]\t: ")).lower()
    rlgen = ['male', 'female']
    if vjenisKelamin not in rlgen:
        raise Exception("Error")

    vumur = int(input("Umur Bulan\t: "))
    bbs = float(input("BeratBadan [Kg]\t: "))
    gigi = str(input("Gigi [Baik/Cukup/Kurang]\t: ")).lower()
    png_kiri = str(input("Penglihatan Kiri [Baik/Kurang]\t: ")).lower()
    png_kanan = str(input("Penglihatan Kanan [Baik/Kurang]\t: ")).lower()
    pd_kiri = str(input("Pendengaran Kiri [Baik/Kurang]\t: ")).lower()
    pd_kanan = str(input("Pendengaran Kanan [Baik/Kurang]\t: ")).lower()
    berbicara = str(input("Berbicara [Baik/Cukup/Kurang]\t: ")).lower()
    
    print("")
    print("=============================================================")
    print("")


    obj1 = BayesGizi(dataframe=sam, tipe_gizi='Gizi Buruk')
    print("Gizi Buruk: \t", obj1.compute(vjenisKelamin, uage=vumur, weight=bbs, gigi=gigi, side=[png_kanan, png_kiri], snd=[pd_kanan, pd_kiri], utalk=berbicara))

    obj2 = BayesGizi(dataframe=sam, tipe_gizi='Gizi Kurang')
    print("Gizi Kurang: \t",obj2.compute(vjenisKelamin, uage=vumur, weight=bbs, gigi=gigi, side=[png_kanan, png_kiri], snd=[pd_kanan, pd_kiri], utalk=berbicara))

    obj3 = BayesGizi(dataframe=sam, tipe_gizi='Gizi Normal')
    print("Gizi Normal: \t",obj3.compute(vjenisKelamin, uage=vumur, weight=bbs, gigi=gigi, side=[png_kanan, png_kiri], snd=[pd_kanan, pd_kiri], utalk=berbicara))

    obj4 = BayesGizi(dataframe=sam, tipe_gizi='Gizi Lebih')
    print("Gizi Lebih: \t",obj4.compute(vjenisKelamin, uage=vumur, weight=bbs, gigi=gigi, side=[png_kanan, png_kiri], snd=[pd_kanan, pd_kiri], utalk=berbicara))
    