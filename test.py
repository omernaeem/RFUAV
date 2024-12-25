import os


def rename_folders(target_directory, old_name, new_name):
    """
    Rename folders in the target directory from old_name to new_name.

    :param target_directory: The directory containing the folders to rename.
    :param old_name: The current name of the folders to rename.
    :param new_name: The new name for the folders.
    """
    try:
        # List all items in the target directory
        for item in os.listdir(target_directory):
            item_path = os.path.join(target_directory, item)
            # Check if the item is a directory and its name matches old_name
            if os.path.isdir(item_path) and item == old_name:
                new_item_path = os.path.join(target_directory, new_name)
                # Rename the directory
                os.rename(item_path, new_item_path)
                print(f'Renamed: {item_path} to {new_item_path}')
    except Exception as e:
        print(f'An error occurred: {e}')


def main():
    # Specify the target directory, old name, and new name
    data_path = 'E:/Drone_dataset/RFUAV/augmentation_exp1_MethodSelect/bechmark_test/'

    snrs = os.listdir(data_path)

    for snr in snrs:
        CMS = os.listdir(os.path.join(data_path, snr))
        for CM in CMS:
            path = os.path.join(data_path, snr, CM)
            names = os.listdir(path)
            for name in names:
                if name == 'DIMINI4PRO-17db-60db_2450m_100m_20-noise':
                    rename_folders(path, name, 'DJIMINI4PRO')

                if name == 'DJFPVCOMBO-28db-90db_5760m_100m_40-noise':
                    rename_folders(path, name, 'DJIFPVCOMBO')

                if name == 'DJI AVTA2-SNR2dB-85db_5760m_100m_20m(1-noise':
                    rename_folders(path, name, 'DJIAVATA2')

                if name == 'DJMAVIC3PRO-16db-90db_5800m_100m_20-noise':
                    rename_folders(path, name, 'DJIMAVIC3PRO')

                if name == 'DJMINI3--46db-60db_2470m_100m_20-noise':
                    rename_folders(path, name, 'DJIMINI3')

        print(f'{CM} Done!')
    print(f'{snr} Done!')


if __name__ == '__main__':
    main()