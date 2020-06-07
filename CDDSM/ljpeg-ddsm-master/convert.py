import os

path_to_ddsm = "/home/www_kmk_com/medical-image-forgery-detection/CDDSM/figment.csee.usf.edu/pub/DDSM/"

for root, subFolders, file_names in os.walk(path_to_ddsm):
    for file_name in file_names:
        if ".LJPEG" in file_name:
            ljpeg_path = os.path.join(root, file_name)
            out_path = os.path.join(root, file_name)
            out_path = out_path.split('.LJPEG')[0] + ".jpg"
            
            cmd = 'python3 ljpeg.py "{0}" "{1}" --visual --scale 1.0'.format(ljpeg_path, out_path)
            os.system(cmd)
            

print('done')