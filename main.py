from corpus import *
from nb import *
from datetime import datetime
import os
import pickle
from tqdm import tqdm


def write_result(result_lines):
    try:
        # create a result folder
        newpath = r'./result'
        if os.path.isdir(newpath):
            pass
        else:
            os.makedirs(newpath)

        # get current date and time
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_file = f'{dt_string}_results.txt'

        # write results to file
        with open(newpath + "/" + output_file, 'w') as f:
            for line in result_lines:
                f.write(line)
                f.write('\n')

        print(f"Result file {output_file} is successfully exported")
        print(f"to {os.path.abspath(newpath)}")

    except Exception as e:
        print(f'Something went wrong, caught {type(e)}: e ')


def do_train(spam_path, ham_path):
    try:
        # preprocessing
        print("Start preprocessing training data. . .")
        spam_email = read_dataset(spam_path)
        ham_email = read_dataset(ham_path)
        emails = spam_email + ham_email

        # start training
        print("")
        print("Start training Spam Filter. . .")
        spam_filter = SpamFilter()
        spam_filter.train(emails)

        # save training object to pickle file
        with open('training_spam.pickle', 'wb') as handle:
            pickle.dump(spam_filter, handle)

        print("Training completed.")
        print("You can now use your spam filter.")

    except FileNotFoundError:
        print(f'No such file or directory. Try again.')
    except Exception as e:
        print(f'Something went wrong, caught {type(e)}: e ')


def classify_single(filepath):
    try:
        # check if training file exist
        if os.path.exists("training_spam.pickle"):
            with open('training_spam.pickle', 'rb') as handle:
                spam_filter = pickle.load(handle)

            # classify one file
            email = read_file(filepath)
            result = spam_filter.classify(email)
            label = result[0]
            spam_score = round(result[1], 4)

            # get filename
            filename = filepath.split("\\")[-1]

            # output result on console
            print(f'Result : {filename} {spam_score} {label}')

        else:
            print("Training file is not found. You have to train a spam filter first.")
    except FileNotFoundError:
        print(f'No such file or directory. Try again.')
    except Exception as e:
        print(f'Something went wrong, caught {type(e)}: e ')



def classify_batch(folderpath):
    try:
        # check if training file exist
        if os.path.exists("training_spam.pickle"):
            with open('training_spam.pickle', 'rb') as handle:
                spam_filter = pickle.load(handle)

                # read files in folder
                file_names = get_paths(folderpath)
                result_lines = []

                print("")
                print("Start classifying. . .")

                for file in tqdm(file_names):

                    fullpath = folderpath + file
                    # read and classify
                    email = read_file(fullpath)
                    result = spam_filter.classify(email)
                    # write result
                    label = result[0]
                    spam_score = round(result[1], 4)
                    output_txt = f'{file} {spam_score} {label}'
                    result_lines.append(output_txt)

            write_result(result_lines)

        else:
            print("Training file is not found. You have to train a spam filter first.")

    except FileNotFoundError:
        print(f'No such file or directory. Try again.')
    except Exception as e:
        print(f'Something went wrong, caught {type(e)}: e ')


if __name__ == "__main__":
    header = """

███████╗██████╗  █████╗ ███╗   ███╗    ███████╗██╗██╗  ████████╗███████╗██████╗
██╔════╝██╔══██╗██╔══██╗████╗ ████║    ██╔════╝██║██║  ╚══██╔══╝██╔════╝██╔══██╗
███████╗██████╔╝███████║██╔████╔██║    █████╗  ██║██║     ██║   █████╗  ██████╔╝
╚════██║██╔═══╝ ██╔══██║██║╚██╔╝██║    ██╔══╝  ██║██║     ██║   ██╔══╝  ██╔══██╗
███████║██║     ██║  ██║██║ ╚═╝ ██║    ██║     ██║███████╗██║   ███████╗██║  ██║
╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝     ╚═╝    ╚═╝     ╚═╝╚══════╝╚═╝   ╚══════╝╚═╝  ╚═╝
                                                        BY : Mattalika Intarahom

°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸
    """
    print(header)
    print(" OPTIONS ".center(80, '.'))
    print("")
    print("Enter [T] to train your data")
    print("Enter [F] to filter your email(s)")
    print("Enter [H] for help")
    print("Enter [E] for exit")
    print("")

    while True:
        option = input(">> Please enter your command: ").lower()

        # train menu
        if option == "t":
            spam_path = input(">> Enter a path to your [spam] training data. (Example: data/train/spam/): ")
            ham_path = input(">> Enter a path to your [ham] training data. (Example: data/train/ham/): ")
            do_train(spam_path, ham_path)

        # filter menu
        elif option == "f":
            classify_option = input("   >> Enter [S] to filter a single email, [B] to filter as a batch: ").lower()
            if classify_option == "s":
                filepath = input("   >> Enter a path to your email file. (Example: data/train/ham/email.txt): ")
                classify_single(filepath)
            elif classify_option == "b":
                folderpath = input("    >> Enter a path to your folder (Example: data/test/ham/): ")
                classify_batch(folderpath)

        # help menu
        elif option == "h":
            print("")
            print(" HOW TO TRAIN ".center(80, '.'))
            print("To classify an email, you will have to train the spam filter first.")
            print("You will have to provide a path to your training data.")
            print("Once you finished the training, you do not have to train it again")
            print("unless you want to train it with different training data.")
            print("")
            print(" HOW TO FILTER ".center(80, '.'))
            print("You can filter email(s) as a single email or as a batch.")
            print("For a single file, you will have to provide a path including your file name.")
            print("The result will be shown on the screen for a single email.")
            print("For a batch you will have to provide a path to your folder containing email files.")
            print("The result will then be exported to a folder as a .txt file.")
            print("")

        # exit program
        elif option == "e":
            print("Program is now closing.")
            exit(0)



