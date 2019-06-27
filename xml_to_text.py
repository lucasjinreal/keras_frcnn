import xml.etree.ElementTree as ET
import os


images_path="./Images/" # Add path of labelled images
xml_path='/Annotations/' # add images xml file path



print("Converting images")
for file in os.listdir(xml_path):

    
    root = ET.parse(xml_path+'/'+file)
    #image_name=root.find("filename").text
    for elem in root.iter():
        
        if 'filename' in elem.tag:
            image_name=elem.text
        if 'object' in elem.tag:
            for attr in list(elem):
                #print(attr)
                if 'name' in attr.tag:
                    object_name=attr.text
                
                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            xmin=dim.text
                        
                        if 'ymin' in dim.tag:
                            ymin=dim.text
                        
                        if 'xmax' in dim.tag:
                            xmax=dim.text
                        
                        if 'ymax' in dim.tag:
                            ymax=dim.text
            data=images_path+image_name+"," +xmin+','+ ymin +',' + xmax + ","+ ymax + ","+ object_name
           
            with open("test.txt", "a") as myfile:
                myfile.write(data+"\n")
print("finished")
    
