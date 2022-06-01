"""
Скрипт запуска алгоритма обнаружения точек бордюра start_detection.py 

"""
import argparse
from pathlib import Path
import os
from time import gmtime, strftime

import numpy as np
import pandas as pd

import laspy
from plyfile import PlyData, PlyElement
import open3d as o3d

#Директория сохранения результирующего файла по умолчанию
DEFAULT_SAVE_PATH = "{0}\\bourders_{1}.las".format(os.path.expanduser("~\\Desktop"), strftime("%Y-%m-%d_%H:%M:%S", gmtime()))


def ply_to_dataframe(path): 
 
		"""Функция чтения .ply файла и преобразования считанных данных к pd.DataFrame
		Parameters
		----------
		path : string
			Путь до .ply файла

		Returns
		-------
		pandas DataFrame с данными .ply файла
		"""

		plydata = PlyData.read(path)		
		np_ar = np.asarray(plydata)
		df = pd.DataFrame(np_ar[0])

		return df

def save_las_file(path, array):
		"""Функция сохранения .las файла
		----------
		path : string
			Путь для сохранения .las файла

		Returns
		-------
		"""

		outfile = laspy.create()
		outfile.X = array[:, 0]
		outfile.Y = array[:, 1]
		outfile.Z = array[:, 2]
		outfile.write('result_1.las')

		return 

def remove_noise(cloud, nb_points = 6, radius = 0.14):
		"""Функция удаления выбросов из облака точек
		----------
		cloud : np.array
			Координаты облака точек XYZ

		nb_points : int 
			Минимальное количество точек, которое должна содержать сфера.

		radius : float
			радиус сферы для подсчета соседних точек

		Returns
		np.array
			Облако точек без выбросов
		-------
		"""
		
		p_cloud = o3d.geometry.PointCloud()
		p_cloud.points = o3d.utility.Vector3dVector(cloud)
		cloud_points_new = p_cloud.remove_radius_outlier(nb_points = nb_points, radius = radius)

		return cloud_points_new[1]

def get_describe_df(df):
		"""Функция расчета описательных статистик интенсивности отраженного лазерного импульса Lidar для каждого класса
		Parameters
		----------
		df : pd.DataFrame

		Returns
		-------
		pd.DataFrame с статистиками для интенсивности
		"""

		df_describe = df.groupby("scalar_Label")["scalar_Intensity"].describe()
		#Вычислим % точек для каждого класса от общего количества
		df_describe ["%_of_all_data"] = round(100 * df_describe["count"]/df.shape[0], 1)
		#Добавим моду
		df_describe ["mode"] = df.groupby("scalar_Label").agg({"scalar_Intensity" :[pd.Series.mode]})

		return df_describe 

def detect_curb_points(input_file, save_path):
		"""Функция детектирования бордюрных точек
		Parameters
		----------
		input_file : string
		save_path : string

		"""
		#Напечатаем описательную статистика по интенсивности отраженного лазерного излучения для текущих данных
		df = ply_to_dataframe(path = input_file)
		print("\nОписательные статистики интенсивности отраженного лазерного импульса Lidar для каждого класса\n\n {0}".format(get_describe_df(df)))

		#Список для результирующих точек бордюра
		curb_points = []

		dtype = [('x', '<f8'), ('y', '<f8'), ('z', '<f8'),\
							 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),\
							 ('scalar_Intensity', '<f4'), ('scalar_GPSTime', '<f4'),\
							 ('scalar_ScanAngleRank', '<f4'), ('scalar_Label', '<f4')]
		dt = np.dtype(dtype)

		#Преобразование к np.array
		dn = df.to_numpy()
		#Сортировка данных по оси y(по направлению движения автомобиля)
		dn = dn[dn[:, 1].argsort()]
		#Исходные данные будут проанализированы частями(parts), размер части 750000 точек
		size_part = 750000
		full_parts = dn.shape[0] // size_part
		small_parts = ((dn.shape[0] % size_part) > 0)
		#Для каждой части исходных данных будем отдельно запускать алгоритм
		#Результат работы алгоритма будем записывать в curb_points 
		for part in range(full_parts + small_parts):
			dn_part = dn[part * size_part : part * size_part + size_part] if part < full_parts else dn[(part - 1) * size_part + size_part : ]
					
			#Применим алгоритм RANSAC для выделения облака точек, относящихся к поверхности земли(отбрасываем точки зданий, столбов, машин и т.д.)
			cloud= o3d.geometry.PointCloud()
			cloud.points = o3d.utility.Vector3dVector(dn_part[:, :3])
			plane_model, inliers = cloud.segment_plane(distance_threshold=0.17,
																							 ransac_n=3,
																							 num_iterations=600)

			#Запишем облако точек, относящихся к земле(проезжая часть, бордюр, пешеходная зона, газон)
			dn_part = dn_part[inliers]


			#В список будем сохранять потенциальные точки бордюра
			potential_curb_points = []

			#Для каждого момента времени испускания импульса(GPS TIME)
			for time_imp in np.unique(dn[:, 7]):
				dn_part_time = dn_part[dn_part[:, 7] == time_imp]
				#Дополнительно сортируем по оси x
				dn_part_time = dn_part_time[dn_part_time[:, 0].argsort()]

				#Для каждого угла сканирования
				for angle_scan in np.unique(dn_part_time[:, 3]):
					dn_part_time_angle = dn_part_time[dn_part_time[:, 3] == angle_scan]
					#Для какждой точки облака с моментом испускания импульса time_imp и углом сканирования angle_scan 
					#находим точки с макс. и мин. высотой(по оси z) среди ближайших 10 точек к исходной
					for point in range(5, len(dn_part_time_angle) - 5):
						local_points = dn_part_time_angle[(point - 5):(point + 5)][:, 2]
						MIN_z = local_points.min()
						MAX_z = local_points.max()
						diff = MAX_z - MIN_z
						#Если разница превышает порог, то точки считается потенциально бордюрной 
						if diff >= 0.12:
							potential_curb_points.append(list(dn_part_time_angle[point]))
			#Сохраним облако точек потенциальных бордюров со всеми параметрами
			pre_result = np.array(potential_curb_points)


			#Из предварительного результата возьмем точки, которые по интенсивности лежат в диапазоне от 17 до 25.
			pre_result  = pre_result[(pre_result [:, 6] <= 25) & (pre_result [:, 6] >=17)]

			#Два раза применим функцию удаления шумов из облака точек, потенциально относящиеся к бордюрам
			
			pre_result = pre_result[remove_noise(cloud = pre_result[:, :3])]
			result = pre_result[remove_noise(cloud = pre_result[:, :3])]

			for point in result:
				curb_points.append(point)

			print("Завершено: {0} / {1}".format(part + 1, full_parts + small_parts))

		curb_points = np.array(curb_points)
			#Сохраним результат в .las файл
		save_las_file(save_path, curb_points)


		#Типы данных для записи np.array в .ply файл
		dtype = [('x', '<f8'), ('y', '<f8'), ('z', '<f8'),\
							 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),\
							 ('scalar_Intensity', '<f4'), ('scalar_GPSTime', '<f4'),\
							 ('scalar_ScanAngleRank', '<f4'), ('scalar_Label', '<f4')]
		dt = np.dtype(dtype)
		g = []
		for point in curb_points:
			g.append(tuple(point))
		answ = np.array([g],dtype = dt)
		el = PlyElement.describe(answ[0], 'some_name')
		PlyData([el]).write('OPAr.ply')


def parse_args():

		"""Функция парсинга входных аргументов"""

		parser = argparse.ArgumentParser()
		parser.add_argument('--input_data', type=str, help='input data(.ply file)')
		parser.add_argument('--save_results', type=str, default=DEFAULT_SAVE_PATH, help='save results in .las format')
		opt = parser.parse_args()

		return opt


if __name__ == "__main__":
		opt = parse_args()
		detect_curb_points(input_file = opt.input_data, save_path = opt.save_results)



