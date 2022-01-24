# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 10:11:32 2019

@author: bmontpetit
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 13:55:58 2018

@author: bmontpetit
"""

class ActivePool(object):
    
    import multiprocess
    
    def __init__(self):
        super(ActivePool, self).__init__()
        self.mgr = multiprocess.Manager()
        self.active = self.mgr.list()
        self.lock = multiprocess.Lock()
    def makeActive(self, name):
        with self.lock:
            self.active.append(name)
    def makeInactive(self, name):
        with self.lock:
            self.active.remove(name)
    def __str__(self):
        with self.lock:
            return str(self.active)
        


def rs2cali(s, pool, zippath, outpath, img, i, numfiles):#, target):    
    
    import subprocess, gdal, zipfile, shutil, os

    import numpy as np
    import xml.etree.ElementTree as ET
    import scipy
    from scipy import signal
    
    nam = os.getpid()
    
    with s:
        pool.makeActive(nam)
        print('Currently running Parent PIDs:' + str(pool))
        
        print('File ' + str(i+1) + ' of ' + str(numfiles) + ' : ' + img)
        
        #Unzip file
        im_zip = zipfile.ZipFile(os.path.join(zippath, img))
        im_zip.extractall(outpath)
        im_zip.close
        
        os.chdir(os.path.join(outpath, img.strip('.zip')))
#        os.chdir(img)
        
        #Read RADARSAT-2 ScanSAR image
        ras = gdal.Open('product.xml')
        x = ras.RasterXSize
        y = ras.RasterYSize
        hh = np.float32(ras.GetRasterBand(1).ReadAsArray())
        hv = np.float32(ras.GetRasterBand(2).ReadAsArray())
        
        #Get the calibration lookup table
        tree = ET.parse(r'lutSigma.xml')
        root = tree.getroot()
        gains = np.float32(np.array(root.find('gains').text.split(' ')))
        
        #Get the incidence angle information from the metadata
        tree = ET.parse(r'product.xml')
        root = tree.getroot()
        pref = root.tag.strip('product')
        nearang = np.float32(root.find(pref + 'imageGenerationParameters').find(pref + 'sarProcessingInformation').find(pref + 'incidenceAngleNearRange').text)
        farang = np.float32(root.find(pref + 'imageGenerationParameters').find(pref + 'sarProcessingInformation').find(pref + 'incidenceAngleFarRange').text)
        look = root.find(pref + 'sourceAttributes').find(pref + 'orbitAndAttitude').find(pref + 'orbitInformation').find(pref + 'passDirection').text
        
        for lut in root.iter(pref + 'referenceNoiseLevel'):

            if lut.attrib['incidenceAngleCorrection'] == 'Sigma Nought':
        
                steps = int(lut.findall('{http://www.rsi.ca/rs2/prod/xml/schemas}stepSize')[0].text)
                first_value = int(lut.findall('{http://www.rsi.ca/rs2/prod/xml/schemas}pixelFirstNoiseValue')[0].text)
                noise = np.array(lut.findall('{http://www.rsi.ca/rs2/prod/xml/schemas}noiseLevelValues')[0].text.split(' '),np.float32)
                
        gains_temp = np.zeros(x, np.float32)
        gains_temp[first_value::steps] = np.power(10, noise/10)
        kernel = signal.triang(2*steps - 1)
        noisepat = 10 * np.log10(scipy.ndimage.filters.convolve(gains_temp, kernel, mode="constant"))
        
        #Set the incidence angle in the right order
        if look == 'Ascending':
            
            incang = np.interp(np.arange(x),[0,len(np.arange(x))-1],[nearang,farang])
            
        else:
                
            incang = np.interp(np.arange(x),[0,len(np.arange(x))-1],[farang,nearang])
        
        #initialize the incidence angle raster
        incangs = np.zeros([y,x], dtype = np.float32)
        noiseimg = np.zeros([y,x], dtype = np.float32)
        
        #Calibrate the HH-HV bands and generates the incidence angle band
        for i in np.arange(y):
                
                hh[i,] = np.true_divide(hh[i,]**2, gains)
                hv[i,] = np.true_divide(hv[i,]**2, gains)
                incangs[i,] = incang
                noiseimg[i,] = noisepat
        
        #Some values are =0 and you can't compute log10(0) so I set a low value
        hh[hh<=0] = 1E-9    
        hv[hv<=0] = 1E-9
        #Convert the values to decibels (dB)
        hh = 10*np.log10(hh)
        hv = 10*np.log10(hv)
        
        os.chdir(outpath)
#        outfile2 = 'calib_' + img.split('\\')[2] + '.tif'
        outfile2 = 'calib_' + img.strip('.zip') + '.tif'
        
        #Create the calibrated image
        driver = gdal.GetDriverByName('GTiff')
        driver.Register()
        outDataset = driver.Create(outfile2, x, y, 4, gdal.GDT_Float32)
        
        #Get the georeference info from original image and set it to new one
        geoTransform = ras.GetGeoTransform()
        outDataset.SetGeoTransform(geoTransform)
        gcps = ras.GetGCPs()
        gcpproj = ras.GetGCPProjection()
        proj = ras.GetProjection()
        outDataset.SetProjection(proj)
        outDataset.SetGCPs(gcps, gcpproj)
        
        #Write HH band
        outhh = outDataset.GetRasterBand(1)
        outhh.WriteArray(hh, 0, 0)
        
        #Write HV band
        outhv = outDataset.GetRasterBand(2)
        outhv.WriteArray(hv, 0, 0)
        
        #Write incidence angle band
        outang = outDataset.GetRasterBand(3)
        outang.WriteArray(incangs, 0, 0)
        
        #Write incidence angle band
        outnoise = outDataset.GetRasterBand(4)
        outnoise.WriteArray(noiseimg, 0, 0)
        
        #Close everything
        outhh = None
        outhv = None
        outang = None
        outnoise = None
        outDataset = None
        driver = None
        proj = None
        geoTransform = None
        gcps = None
        gcpproj = None
        ras = None
        
        del outhh, outhv, outang, outnoise, outDataset, driver, proj, geoTransform
        
        #This part clips out the land to keep only marine area, I don't use coastal zones either in my training data.
        outfile3 = 'S0_' + img.strip('.zip') + '.tif'
#        outfile3 = img.split('\\')[2] + '.tif'
        
#        fullsizepath = r'G:\\' + target   
#        fullsizepath = r'E:\ScanSARclip2'   
        
        #Clip only the marine area and remove all land *** this is optional
        subprocess.call('gdalwarp -t_srs EPSG:6931 -srcnodata None -dstnodata NaN -overwrite -cutline Data\Coronation_simple.shp -crop_to_cutline -tr 50 50 -tap ' + outfile2 + ' ' + outfile3)
#        subprocess.call('gdalwarp -t_srs EPSG:6931 -srcnodata None -dstnodata NaN -overwrite -cutline G:\Shapefile\CoronationHR.shp -crop_to_cutline -tr 50 50 -tap ' + outfile2 + ' ' + os.path.join(fullsizepath, outfile3))
        
        os.remove(outfile2)
        
        del outfile2, ras, gains, root, tree, hh, hv, x, y, gcps, gcpproj, nearang, farang, incang, incangs, look
        
        shutil.rmtree(os.path.join(outpath, img.strip('.zip')))
        
        pool.makeInactive(nam)
        print('Currently running Parent PIDs:' + str(pool))
    
if __name__ == "__main__":
    
    
    import multiprocess
    import numpy as np
    from os.path import join
    from os import listdir
    
    target = ''
    
    indir = join(r'Data', target)
    outpath = join(r'Data', target)

    files = [f for f in listdir(indir) if f.endswith('.zip')]    
    
    available_cpus = 5
    pool = ActivePool()
    s = multiprocess.Semaphore(available_cpus)
    
    joblist = []
    
    for i in np.arange(len(files)):
        joblist.append(multiprocess.Process(target = rs2cali, args = (s, pool, indir, outpath, files[i], i, len(files))))
        
    for job in joblist:
        job.start()
    
    for job in joblist:
        job.join()