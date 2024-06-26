import os
from detector_plagio import detectar_plagio

current_directory = os.getcwd()

# Lista de carpetas a analizar
Z1_Z1 = os.path.join(current_directory, "Dataset_C/A2016/Z1/Z1/")
Z1_Z2 = os.path.join(current_directory, "Dataset_C/A2016/Z1/Z2/")
Z1_Z3 = os.path.join(current_directory, "Dataset_C/A2016/Z1/Z3/")
Z1_Z4 = os.path.join(current_directory, "Dataset_C/A2016/Z1/Z4/")

Z2_Z1 = os.path.join(current_directory, "Dataset_C/A2016/Z2/Z1/")
Z2_Z2 = os.path.join(current_directory, "Dataset_C/A2016/Z2/Z2/")
Z2_Z3 = os.path.join(current_directory, "Dataset_C/A2016/Z2/Z3/")
Z2_Z4 = os.path.join(current_directory, "Dataset_C/A2016/Z2/Z4/")

#Lista de pares de archivos plagiados
archivos = [
    # Carpeta 1 - A2016/Z1/Z1
    (os.path.join(Z1_Z1, "student7386.c"), os.path.join(Z1_Z1, "student5378.c")),
    (os.path.join(Z1_Z1, "student2821.c"), os.path.join(Z1_Z1, "student8295.c")),
    (os.path.join(Z1_Z1, "student4934.c"), os.path.join(Z1_Z1, "student6617.c")),
    (os.path.join(Z1_Z1, "student8598.c"), os.path.join(Z1_Z1, "student3331.c")),
    (os.path.join(Z1_Z1, "student7888.c"), os.path.join(Z1_Z1, "student7704.c")),
    (os.path.join(Z1_Z1, "student4959.c"), os.path.join(Z1_Z1, "student1482.c")),
    (os.path.join(Z1_Z1, "student2939.c"), os.path.join(Z1_Z1, "student9949.c")),
    (os.path.join(Z1_Z1, "student5512.c"), os.path.join(Z1_Z1, "student4852.c")),
    (os.path.join(Z1_Z1, "student9160.c"), os.path.join(Z1_Z1, "student7443.c")),
    (os.path.join(Z1_Z1, "student8357.c"), os.path.join(Z1_Z1, "student6877.c")),
    (os.path.join(Z1_Z1, "student2086.c"), os.path.join(Z1_Z1, "student7173.c")),
    (os.path.join(Z1_Z1, "student5789.c"), os.path.join(Z1_Z1, "student5673.c")),
    (os.path.join(Z1_Z1, "student9358.c"), os.path.join(Z1_Z1, "student2953.c")),
    (os.path.join(Z1_Z1, "student9498.c"), os.path.join(Z1_Z1, "student8796.c")),
    (os.path.join(Z1_Z1, "student6776.c"), os.path.join(Z1_Z1, "student9805.c")),
   
    # Carpeta 2 - A2016/Z1/Z2
    (os.path.join(Z1_Z2, "student4124.c"), os.path.join(Z1_Z2, "student9538.c")),
    (os.path.join(Z1_Z2, "student8598.c"), os.path.join(Z1_Z2, "student3331.c")),
    (os.path.join(Z1_Z2, "student9498.c"), os.path.join(Z1_Z2, "student8796.c")),
    (os.path.join(Z1_Z2, "student7888.c"), os.path.join(Z1_Z2, "student7704.c")),
    (os.path.join(Z1_Z2, "student6534.c"), os.path.join(Z1_Z2, "student5381.c")),
    (os.path.join(Z1_Z2, "student4959.c"), os.path.join(Z1_Z2, "student1482.c")),
    (os.path.join(Z1_Z2, "student2939.c"), os.path.join(Z1_Z2, "student9949.c")),
    (os.path.join(Z1_Z2, "student2821.c"), os.path.join(Z1_Z2, "student4155.c")),
    (os.path.join(Z1_Z2, "student5512.c"), os.path.join(Z1_Z2, "student4852.c")),
    (os.path.join(Z1_Z2, "student1326.c"), os.path.join(Z1_Z2, "student1624.c")),
    (os.path.join(Z1_Z2, "student8187.c"), os.path.join(Z1_Z2, "student3631.c")),
    (os.path.join(Z1_Z2, "student6054.c"), os.path.join(Z1_Z2, "student7341.c")),
    (os.path.join(Z1_Z2, "student9160.c"), os.path.join(Z1_Z2, "student7443.c")),
    (os.path.join(Z1_Z2, "student1120.c"), os.path.join(Z1_Z2, "student5226.c")),

    # Carpeta 3 - A2016/Z1/Z3
    (os.path.join(Z1_Z3, "student7386.c"), os.path.join(Z1_Z3, "student5378.c")),
    (os.path.join(Z1_Z3, "student2821.c"), os.path.join(Z1_Z3, "student8295.c")),
    (os.path.join(Z1_Z3, "student4155.c"), os.path.join(Z1_Z3, "student2197.c")),
    (os.path.join(Z1_Z3, "student8598.c"), os.path.join(Z1_Z3, "student3331.c")),
    (os.path.join(Z1_Z3, "student5573.c"), os.path.join(Z1_Z3, "student9498.c")),
    (os.path.join(Z1_Z3, "student4420.c"), os.path.join(Z1_Z3, "student6617.c")),
    (os.path.join(Z1_Z3, "student4059.c"), os.path.join(Z1_Z3, "student7704.c")),
    (os.path.join(Z1_Z3, "student6534.c"), os.path.join(Z1_Z3, "student4934.c")),
    (os.path.join(Z1_Z3, "student2925.c"), os.path.join(Z1_Z3, "student4280.c")),
    (os.path.join(Z1_Z3, "student4959.c"), os.path.join(Z1_Z3, "student1482.c")),
    (os.path.join(Z1_Z3, "student2939.c"), os.path.join(Z1_Z3, "student9949.c")),
    (os.path.join(Z1_Z3, "student5512.c"), os.path.join(Z1_Z3, "student4852.c")),
    (os.path.join(Z1_Z3, "student1326.c"), os.path.join(Z1_Z3, "student3560.c")),
    (os.path.join(Z1_Z3, "student4863.c"), os.path.join(Z1_Z3, "student5380.c")),
    (os.path.join(Z1_Z3, "student9821.c"), os.path.join(Z1_Z3, "student8133.c")),
    (os.path.join(Z1_Z3, "student3424.c"), os.path.join(Z1_Z3, "student3756.c")),
    (os.path.join(Z1_Z3, "student7495.c"), os.path.join(Z1_Z3, "student5741.c")),
    (os.path.join(Z1_Z3, "student1466.c"), os.path.join(Z1_Z3, "student1266.c")),
    (os.path.join(Z1_Z3, "student6054.c"), os.path.join(Z1_Z3, "student7341.c")),
    (os.path.join(Z1_Z3, "student9805.c"), os.path.join(Z1_Z3, "student8357.c")),
    (os.path.join(Z1_Z3, "student5581.c"), os.path.join(Z1_Z3, "student8089.c")),
    (os.path.join(Z1_Z3, "student3350.c"), os.path.join(Z1_Z3, "student4226.c")),
    (os.path.join(Z1_Z3, "student2831.c"), os.path.join(Z1_Z3, "student4343.c")),
    (os.path.join(Z1_Z3, "student3567.c"), os.path.join(Z1_Z3, "student1453.c")),
    (os.path.join(Z1_Z3, "student5170.c"), os.path.join(Z1_Z3, "student8540.c")),

    # Carpeta 4 - A2016/Z1/Z4
    (os.path.join(Z1_Z4, "student2821.c"), os.path.join(Z1_Z4, "student8295.c")),
    (os.path.join(Z1_Z4, "student8598.c"), os.path.join(Z1_Z4, "student3331.c")),
    (os.path.join(Z1_Z4, "student7888.c"), os.path.join(Z1_Z4, "student7704.c")),
    (os.path.join(Z1_Z4, "student6534.c"), os.path.join(Z1_Z4, "student6617.c")),
    (os.path.join(Z1_Z4, "student2925.c"), os.path.join(Z1_Z4, "student5381.c")),
    (os.path.join(Z1_Z4, "student4959.c"), os.path.join(Z1_Z4, "student1482.c")),
    (os.path.join(Z1_Z4, "student2939.c"), os.path.join(Z1_Z4, "student9949.c")),
    (os.path.join(Z1_Z4, "student4124.c"), os.path.join(Z1_Z4, "student9538.c")),
    (os.path.join(Z1_Z4, "student5512.c"), os.path.join(Z1_Z4, "student4852.c")),
    (os.path.join(Z1_Z4, "student1266.c"), os.path.join(Z1_Z4, "student9931.c")),
    (os.path.join(Z1_Z4, "student3424.c"), os.path.join(Z1_Z4, "student1845.c")),
    (os.path.join(Z1_Z4, "student6054.c"), os.path.join(Z1_Z4, "student7341.c")),
    (os.path.join(Z1_Z4, "student6776.c"), os.path.join(Z1_Z4, "student1483.c")),
    (os.path.join(Z1_Z4, "student8357.c"), os.path.join(Z1_Z4, "student4934.c")),
    (os.path.join(Z1_Z4, "student7123.c"), os.path.join(Z1_Z4, "student2142.c")),
    (os.path.join(Z1_Z4, "student3671.c"), os.path.join(Z1_Z4, "student7802.c")),
    (os.path.join(Z1_Z4, "student9160.c"), os.path.join(Z1_Z4, "student7443.c")),
    (os.path.join(Z1_Z4, "student1188.c"), os.path.join(Z1_Z4, "student8199.c")),
    (os.path.join(Z1_Z4, "student2887.c"), os.path.join(Z1_Z4, "student9805.c")),
    (os.path.join(Z1_Z4, "student2806.c"), os.path.join(Z1_Z4, "student4108.c")),
    (os.path.join(Z1_Z4, "student5899.c"), os.path.join(Z1_Z4, "student7255.c")),
    (os.path.join(Z1_Z4, "student3856.c"), os.path.join(Z1_Z4, "student4100.c")),

    #Carpeta 5 - A2016/Z2/Z1
    (os.path.join(Z2_Z1, "student7386.c"), os.path.join(Z2_Z1, "student5380.c")),
    (os.path.join(Z2_Z1, "student8192.c"), os.path.join(Z2_Z1, "student2736.c")),
    (os.path.join(Z2_Z1, "student1616.c"), os.path.join(Z2_Z1, "student8886.c")),
    (os.path.join(Z2_Z1, "student3421.c"), os.path.join(Z2_Z1, "student9296.c")),
    (os.path.join(Z2_Z1, "student5612.c"), os.path.join(Z2_Z1, "student5581.c")),
    (os.path.join(Z2_Z1, "student7956.c"), os.path.join(Z2_Z1, "student7697.c")),
    (os.path.join(Z2_Z1, "student2925.c"), os.path.join(Z2_Z1, "student2821.c")),
    (os.path.join(Z2_Z1, "student9949.c"), os.path.join(Z2_Z1, "student2939.c")),
    (os.path.join(Z2_Z1, "student3717.c"), os.path.join(Z2_Z1, "student2508.c")),
    (os.path.join(Z2_Z1, "student3425.c"), os.path.join(Z2_Z1, "student1639.c")),
    (os.path.join(Z2_Z1, "student9160.c"), os.path.join(Z2_Z1, "student2526.c")),
    (os.path.join(Z2_Z1, "student1120.c"), os.path.join(Z2_Z1, "student5624.c")),
    (os.path.join(Z2_Z1, "student7888.c"), os.path.join(Z2_Z1, "student6617.c")),
    (os.path.join(Z2_Z1, "student8598.c"), os.path.join(Z2_Z1, "student5573.c")),
    (os.path.join(Z2_Z1, "student9931.c"), os.path.join(Z2_Z1, "student5961.c")),
    (os.path.join(Z2_Z1, "student2437.c"), os.path.join(Z2_Z1, "student5636.c")),
    (os.path.join(Z2_Z1, "student8199.c"), os.path.join(Z2_Z1, "student4100.c")),

    #Carpeta 6 - A2016/Z2/Z2
    (os.path.join(Z2_Z2, "student7386.c"), os.path.join(Z2_Z2, "student6999.c")),
    (os.path.join(Z2_Z2, "student8192.c"), os.path.join(Z2_Z2, "student2736.c")),
    (os.path.join(Z2_Z2, "student9538.c"), os.path.join(Z2_Z2, "student3631.c")),
    (os.path.join(Z2_Z2, "student5581.c"), os.path.join(Z2_Z2, "student5612.c")),
    (os.path.join(Z2_Z2, "student2406.c"), os.path.join(Z2_Z2, "student8067.c")),
    (os.path.join(Z2_Z2, "student1616.c"), os.path.join(Z2_Z2, "student8886.c")),
    (os.path.join(Z2_Z2, "student3421.c"), os.path.join(Z2_Z2, "student9296.c")),
    (os.path.join(Z2_Z2, "student8796.c"), os.path.join(Z2_Z2, "student8598.c")),
    (os.path.join(Z2_Z2, "student7956.c"), os.path.join(Z2_Z2, "student4043.c")),
    (os.path.join(Z2_Z2, "student3717.c"), os.path.join(Z2_Z2, "student2508.c")),
    (os.path.join(Z2_Z2, "student3425.c"), os.path.join(Z2_Z2, "student1639.c")),
    (os.path.join(Z2_Z2, "student3315.c"), os.path.join(Z2_Z2, "student1453.c")),
    (os.path.join(Z2_Z2, "student1120.c"), os.path.join(Z2_Z2, "student5624.c")),
    (os.path.join(Z2_Z2, "student2513.c"), os.path.join(Z2_Z2, "student6364.c")),
    (os.path.join(Z2_Z2, "student7888.c"), os.path.join(Z2_Z2, "student5660.c")),
    (os.path.join(Z2_Z2, "student9931.c"), os.path.join(Z2_Z2, "student5961.c")),
    (os.path.join(Z2_Z2, "student2160.c"), os.path.join(Z2_Z2, "student6960.c")),
    (os.path.join(Z2_Z2, "student6594.c"), os.path.join(Z2_Z2, "student4682.c")),
    (os.path.join(Z2_Z2, "student4934.c"), os.path.join(Z2_Z2, "student3386.c")),
    (os.path.join(Z2_Z2, "student1772.c"), os.path.join(Z2_Z2, "student5268.c")),
    (os.path.join(Z2_Z2, "student1237.c"), os.path.join(Z2_Z2, "student4100.c")),
    (os.path.join(Z2_Z2, "student2953.c"), os.path.join(Z2_Z2, "student1649.c")),
    (os.path.join(Z2_Z2, "student1738.c"), os.path.join(Z2_Z2, "student9569.c")),
    (os.path.join(Z2_Z2, "student4185.c"), os.path.join(Z2_Z2, "student2585.c")),
    (os.path.join(Z2_Z2, "student5298.c"), os.path.join(Z2_Z2, "student1200.c")),
    (os.path.join(Z2_Z2, "student4322.c"), os.path.join(Z2_Z2, "student6010.c")),
    (os.path.join(Z2_Z2, "student3136.c"), os.path.join(Z2_Z2, "student6705.c")),
    (os.path.join(Z2_Z2, "student2157.c"), os.path.join(Z2_Z2, "student6516.c")),
    (os.path.join(Z2_Z2, "student5611.c"), os.path.join(Z2_Z2, "student2967.c")),

    #Carpeta 7 - A2016/Z2/Z3
    (os.path.join(Z2_Z3, "student7386.c"), os.path.join(Z2_Z3, "student7888.c")),
    (os.path.join(Z2_Z3, "student8192.c"), os.path.join(Z2_Z3, "student8957.c")),
    (os.path.join(Z2_Z3, "student1845.c"), os.path.join(Z2_Z3, "student3594.c")),
    (os.path.join(Z2_Z3, "student3560.c"), os.path.join(Z2_Z3, "student9473.c")),
    (os.path.join(Z2_Z3, "student8069.c"), os.path.join(Z2_Z3, "student2526.c")),
    (os.path.join(Z2_Z3, "student3421.c"), os.path.join(Z2_Z3, "student9160.c")),
    (os.path.join(Z2_Z3, "student7255.c"), os.path.join(Z2_Z3, "student8089.c")),
    (os.path.join(Z2_Z3, "student4420.c"), os.path.join(Z2_Z3, "student8043.c")),
    (os.path.join(Z2_Z3, "student1616.c"), os.path.join(Z2_Z3, "student8886.c")),
    (os.path.join(Z2_Z3, "student6743.c"), os.path.join(Z2_Z3, "student5789.c")),
    (os.path.join(Z2_Z3, "student1477.c"), os.path.join(Z2_Z3, "student4082.c")),
    (os.path.join(Z2_Z3, "student8796.c"), os.path.join(Z2_Z3, "student5573.c")),
    (os.path.join(Z2_Z3, "student5612.c"), os.path.join(Z2_Z3, "student5581.c")),
    (os.path.join(Z2_Z3, "student4949.c"), os.path.join(Z2_Z3, "student1482.c")),
    (os.path.join(Z2_Z3, "student8753.c"), os.path.join(Z2_Z3, "student8520.c")),
    (os.path.join(Z2_Z3, "student3425.c"), os.path.join(Z2_Z3, "student1639.c")),
    (os.path.join(Z2_Z3, "student6054.c"), os.path.join(Z2_Z3, "student5380.c")),
    (os.path.join(Z2_Z3, "student3315.c"), os.path.join(Z2_Z3, "student1453.c")),
    (os.path.join(Z2_Z3, "student1120.c"), os.path.join(Z2_Z3, "student5624.c")),
    (os.path.join(Z2_Z3, "student2513.c"), os.path.join(Z2_Z3, "student6364.c")),
    (os.path.join(Z2_Z3, "student9931.c"), os.path.join(Z2_Z3, "student5961.c")),
    (os.path.join(Z2_Z3, "student8199.c"), os.path.join(Z2_Z3, "student4100.c")),
    (os.path.join(Z2_Z3, "student5512.c"), os.path.join(Z2_Z3, "student8864.c")),
    (os.path.join(Z2_Z3, "student6594.c"), os.path.join(Z2_Z3, "student4682.c")),
    (os.path.join(Z2_Z3, "student4770.c"), os.path.join(Z2_Z3, "student5899.c")),
    (os.path.join(Z2_Z3, "student9498.c"), os.path.join(Z2_Z3, "student7911.c")),
    (os.path.join(Z2_Z3, "student2953.c"), os.path.join(Z2_Z3, "student1649.c")),
    (os.path.join(Z2_Z3, "student1738.c"), os.path.join(Z2_Z3, "student9569.c")),
    (os.path.join(Z2_Z3, "student5298.c"), os.path.join(Z2_Z3, "student1200.c")),
    (os.path.join(Z2_Z3, "student2508.c"), os.path.join(Z2_Z3, "student1313.c")),
    (os.path.join(Z2_Z3, "student4322.c"), os.path.join(Z2_Z3, "student7616.c")),
    (os.path.join(Z2_Z3, "student4108.c"), os.path.join(Z2_Z3, "student7173.c")),
    (os.path.join(Z2_Z3, "student4554.c"), os.path.join(Z2_Z3, "student8560.c")),
    (os.path.join(Z2_Z3, "student4473.c"), os.path.join(Z2_Z3, "student6516.c")),

    #Carpeta 8 - A2016/Z2/Z4
    (os.path.join(Z2_Z4, "student2956.c"), os.path.join(Z2_Z4, "student5284.c")),
    (os.path.join(Z2_Z4, "student1616.c"), os.path.join(Z2_Z4, "student8886.c")),
    (os.path.join(Z2_Z4, "student3421.c"), os.path.join(Z2_Z4, "student4059.c")),
    (os.path.join(Z2_Z4, "student3756.c"), os.path.join(Z2_Z4, "student8598.c")),
    (os.path.join(Z2_Z4, "student8796.c"), os.path.join(Z2_Z4, "student7888.c")),
    (os.path.join(Z2_Z4, "student5573.c"), os.path.join(Z2_Z4, "student4420.c")),
    (os.path.join(Z2_Z4, "student5612.c"), os.path.join(Z2_Z4, "student3631.c")),
    (os.path.join(Z2_Z4, "student6054.c"), os.path.join(Z2_Z4, "student5380.c")),
    (os.path.join(Z2_Z4, "student1120.c"), os.path.join(Z2_Z4, "student5624.c")),
    (os.path.join(Z2_Z4, "student2513.c"), os.path.join(Z2_Z4, "student6364.c")),
    (os.path.join(Z2_Z4, "student2508.c"), os.path.join(Z2_Z4, "student2340.c")),
    (os.path.join(Z2_Z4, "student8199.c"), os.path.join(Z2_Z4, "student4100.c")),
    (os.path.join(Z2_Z4, "student9476.c"), os.path.join(Z2_Z4, "student7180.c")),
    (os.path.join(Z2_Z4, "student5298.c"), os.path.join(Z2_Z4, "student1200.c")),
    (os.path.join(Z2_Z4, "student5355.c"), os.path.join(Z2_Z4, "student4644.c")),
    (os.path.join(Z2_Z4, "student6743.c"), os.path.join(Z2_Z4, "student5263.c")),
    (os.path.join(Z2_Z4, "student1029.c"), os.path.join(Z2_Z4, "student2585.c")),
    (os.path.join(Z2_Z4, "student6877.c"), os.path.join(Z2_Z4, "student8357.c")),
    (os.path.join(Z2_Z4, "student5659.c"), os.path.join(Z2_Z4, "student6516.c")),
    (os.path.join(Z2_Z4, "student5829.c"), os.path.join(Z2_Z4, "student4415.c")),
    (os.path.join(Z2_Z4, "student5611.c"), os.path.join(Z2_Z4, "student2967.c")),
    (os.path.join(Z2_Z4, "student8892.c"), os.path.join(Z2_Z4, "student7678.c")),

]

no_plagio = [
    # A2016/Z1/Z1 vs A2016/Z2/Z1
    (os.path.join(Z1_Z1, "student1013.c"), os.path.join(Z2_Z1, "student2335.c")),
    (os.path.join(Z1_Z1, "student1263.c"), os.path.join(Z2_Z1, "student2371.c")),
    (os.path.join(Z1_Z1, "student1494.c"), os.path.join(Z2_Z1, "student3610.c")),
    (os.path.join(Z1_Z1, "student2086.c"), os.path.join(Z2_Z1, "student4852.c")),
    (os.path.join(Z1_Z1, "student2111.c"), os.path.join(Z2_Z1, "student5659.c")),
    (os.path.join(Z1_Z1, "student2142.c"), os.path.join(Z2_Z1, "student6029.c")),
    (os.path.join(Z1_Z1, "student2340.c"), os.path.join(Z2_Z1, "student6042.c")),
    (os.path.join(Z1_Z1, "student3856.c"), os.path.join(Z2_Z1, "student4526.c")),
    (os.path.join(Z1_Z1, "student4647.c"), os.path.join(Z2_Z1, "student7258.c")),
    (os.path.join(Z1_Z1, "student5713.c"), os.path.join(Z2_Z1, "student8085.c")),
   
    # A2016/Z1/Z2 vs A2016/Z2/Z4
    (os.path.join(Z1_Z2, "student1013.c"), os.path.join(Z2_Z4, "student1616.c")),
    (os.path.join(Z1_Z2, "student1477.c"), os.path.join(Z2_Z4, "student2421.c")),
    (os.path.join(Z1_Z2, "student2351.c"), os.path.join(Z2_Z4, "student3415.c")),
    (os.path.join(Z1_Z2, "student4128.c"), os.path.join(Z2_Z4, "student3535.c")),
    (os.path.join(Z1_Z2, "student4659.c"), os.path.join(Z2_Z4, "student4628.c")),
    (os.path.join(Z1_Z2, "student5624.c"), os.path.join(Z2_Z4, "student5412.c")),
    (os.path.join(Z1_Z2, "student6410.c"), os.path.join(Z2_Z4, "student6550.c")),
    (os.path.join(Z1_Z2, "student7320.c"), os.path.join(Z2_Z4, "student7507.c")),
    (os.path.join(Z1_Z2, "student8133.c"), os.path.join(Z2_Z4, "student8139.c")),
    (os.path.join(Z1_Z2, "student9095.c"), os.path.join(Z2_Z4, "student9385.c")),

    # A2016/Z1/Z4 vs A2016/Z2/Z3
    (os.path.join(Z1_Z4, "student1981.c"), os.path.join(Z2_Z3, "student2213.c")),
    (os.path.join(Z1_Z4, "student2111.c"), os.path.join(Z2_Z3, "student3133.c")),
    (os.path.join(Z1_Z4, "student3900.c"), os.path.join(Z2_Z3, "student3790.c")),
    (os.path.join(Z1_Z4, "student3610.c"), os.path.join(Z2_Z3, "student4628.c")),
    (os.path.join(Z1_Z4, "student8264.c"), os.path.join(Z2_Z3, "student5581.c")),
    (os.path.join(Z1_Z4, "student8317.c"), os.path.join(Z2_Z3, "student5636.c")),
    (os.path.join(Z1_Z4, "student9676.c"), os.path.join(Z2_Z3, "student6604.c")),
    (os.path.join(Z1_Z4, "student9972.c"), os.path.join(Z2_Z3, "student8006.c")),
    (os.path.join(Z1_Z4, "student7258.c"), os.path.join(Z2_Z3, "student8029.c")),
    (os.path.join(Z1_Z4, "student7193.c"), os.path.join(Z2_Z3, "student1269.c")),

    # A2016/Z1/Z3 vs A2016/Z2/Z2
    (os.path.join(Z1_Z3, "student2247.c"), os.path.join(Z2_Z2, "student9998.c")),
    (os.path.join(Z1_Z3, "student2967.c"), os.path.join(Z2_Z2, "student8780.c")),
    (os.path.join(Z1_Z3, "student4059.c"), os.path.join(Z2_Z2, "student8029.c")),
    (os.path.join(Z1_Z3, "student3717.c"), os.path.join(Z2_Z2, "student7425.c")),
    (os.path.join(Z1_Z3, "student5122.c"), os.path.join(Z2_Z2, "student6960.c")),
    (os.path.join(Z1_Z3, "student6913.c"), os.path.join(Z2_Z2, "student6516.c")),
    (os.path.join(Z1_Z3, "student8192.c"), os.path.join(Z2_Z2, "student5863.c")),
    (os.path.join(Z1_Z3, "student8043.c"), os.path.join(Z2_Z2, "student5172.c")),
    (os.path.join(Z1_Z3, "student8964.c"), os.path.join(Z2_Z2, "student4150.c")),
    (os.path.join(Z1_Z3, "student4888.c"), os.path.join(Z2_Z2, "student2925.c")),
    (os.path.join(Z1_Z3, "student2821.c"), os.path.join(Z2_Z2, "student1981.c")),

    # A2016/Z2/Z1 vs A2016/Z2/Z4
    (os.path.join(Z2_Z1, "student1938.c"), os.path.join(Z2_Z4, "student9972.c")),
    (os.path.join(Z2_Z1, "student1624.c"), os.path.join(Z2_Z4, "student8665.c")),
    (os.path.join(Z2_Z1, "student2675.c"), os.path.join(Z2_Z4, "student7755.c")),
    (os.path.join(Z2_Z1, "student3411.c"), os.path.join(Z2_Z4, "student7085.c")),
    (os.path.join(Z2_Z1, "student4412.c"), os.path.join(Z2_Z4, "student5789.c")),
    (os.path.join(Z2_Z1, "student4959.c"), os.path.join(Z2_Z4, "student4863.c")),
    (os.path.join(Z2_Z1, "student5660.c"), os.path.join(Z2_Z4, "student3841.c")),
    (os.path.join(Z2_Z1, "student6357.c"), os.path.join(Z2_Z4, "student2941.c")),
    (os.path.join(Z2_Z1, "student9029.c"), os.path.join(Z2_Z4, "student2477.c")),
    (os.path.join(Z2_Z1, "student9601.c"), os.path.join(Z2_Z4, "student1649.c")),

]

#Llamo a kernel.py para ver la accuracy total
detectar_plagio(archivos)