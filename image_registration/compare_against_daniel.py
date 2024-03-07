import glob

# daniels_answer_files = glob.glob('./red_daniels_answers/Red Output/*.txt')
# my_answer_files = glob.glob('./answers/Red*.txt')

daniels_answer_files = glob.glob('./test/Blue Output/Blue*.txt')
my_answer_files = glob.glob('./answers/Blue*.txt')

#sort the files so they are in the same order
daniels_answer_files.sort()
my_answer_files.sort()

for daniels_file, my_file in zip(daniels_answer_files, my_answer_files):
    with open(daniels_file, 'r') as daniels, open(my_file, 'r') as mine:
        daniels_answers = daniels.read().split('\n')
        my_answers = mine.read().split('\n')

        if daniels_answers == my_answers:
            print(f'{daniels_file} and {my_file} are the same')
        else:
            print(f'{daniels_file} and {my_file} are different')
            # print(f'Daniels: \n{daniels_answers}')
            # print(f'Mine: \n{my_answers}')
            # print()

            #where the differences are
            for i, (d, m) in enumerate(zip(daniels_answers, my_answers)):
                if d != m:
                    print(f'Question {i+1}: Daniels: {d}, Mine: {m}')
