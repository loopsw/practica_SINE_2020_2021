Práctica utilizando dataset de DEtection of TOXicity in comments In Spanish (DETOXIS)
Nombre: Jackson F. Reyes Bermeo
email: jackson.reyes@hotmail.com


1. Nombre del script: practica_SINE_2020_2021.py

2. Función main: Este es el contenido del main del fichero, 
		 se debe descomentar cada apartado para probar los resultados mostrados
		 en la memoria de esta práctica.
		 El path debe configurar la ruta donde se encuentra el fichero .csv correspondiente
		 
3. Ejecución: Descomentar el aparartado de interes y mandar a ajecutar todo el fichero.

4. Contenido:

    # modelo básico
    #path = "E:/SINE_Pract_2020_2021/dataset/train.csv"
    #processDetoxisTraining(path)
    
    
   # primera modificación
    #path = "E:/SINE_Pract_2020_2021/dataset/train.csv"
    #primerAjuste(path)
    
    # segunda modificación
    #path = "E:/SINE_Pract_2020_2021/dataset/train.csv"
    #segundoAjuste(path)
    
    # tercera modificación
    #path = "E:/SINE_Pract_2020_2021/dataset/train.csv"
    #tercerAjuste(path)

    # Cuarta modificación
    #path = "E:/SINE_Pract_2020_2021/dataset/train.csv"
    #cuartoAjuste(path)
    
    # Evaluacion de 10 folds Crsoss-Fold Validation
    path = "E:/SINE_Pract_2020_2021/dataset/train.csv"
    modeloSeleccionado(path)

    # Predecir el corpus de testing
    path_train = "E:/SINE_Pract_2020_2021/dataset/train.csv"
    path_test = "E:/SINE_Pract_2020_2021/dataset/test.csv"
    run0(path_train,path_test)

NOTAS:  Cada apartado esta definido por una funcion de Python, para algunas pruebas hay que comentar
	y descomentar ciertas lineas que se encuentran en el interior de la funcion de acuerdo al 
	apartado que se quiera reproducir

contacto:


	