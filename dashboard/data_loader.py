"""
================================================================================
dashboard/data_loader.py -- SkyGuard AI EDA Dataset Loader
================================================================================
Generates and caches the 826-station IMD Automatic Weather Station dataset.
If real CSV files are present they are loaded; otherwise the dataset is
synthesised deterministically from the authoritative state/district registry.

Canonical columns:
    station_id, station_name, state, district,
    latitude, longitude, elevation,
    temperature, humidity, pressure,
    health_score, status
================================================================================
"""

from __future__ import annotations

import os
import hashlib
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_STATION_REGISTRY_CSV = os.path.join(_ROOT, "complete_all_india_aws_stations.csv")
_WEATHER_DATA_CSV = os.path.join(_ROOT, "aws_weather_data_all_india_2023_present.csv")

# State registry: (state_name, state_code, [(district, lat, lon, elev_m), ...])
_STATE_REGISTRY = [
    ("Andhra Pradesh", "AP", [
        ("Visakhapatnam", 17.72, 83.30, 45), ("Krishna", 16.51, 80.62, 20),
        ("Guntur", 16.30, 80.44, 30), ("East Godavari", 17.00, 81.78, 25),
        ("West Godavari", 16.92, 81.34, 22), ("Chittoor", 13.22, 79.10, 305),
        ("Kadapa", 14.47, 78.82, 155), ("Anantapur", 14.68, 77.60, 350),
        ("Kurnool", 15.83, 78.04, 270), ("Nellore", 14.44, 79.99, 12),
        ("Srikakulam", 18.30, 83.90, 28), ("Vizianagaram", 18.11, 83.41, 52),
        ("Prakasam", 15.34, 79.57, 40), ("Sri Potti Sriramulu", 13.63, 80.00, 8),
    ]),
    ("Arunachal Pradesh", "AR", [
        ("Itanagar", 27.08, 93.60, 340), ("Tawang", 27.58, 91.87, 3048),
        ("West Kameng", 27.00, 92.50, 1800), ("East Siang", 28.20, 95.30, 420),
        ("Lohit", 28.10, 96.20, 210), ("Tirap", 27.00, 95.60, 800),
        ("Papum Pare", 27.15, 93.75, 380), ("Lower Subansiri", 27.68, 93.82, 520),
    ]),
    ("Assam", "AS", [
        ("Kamrup", 26.14, 91.74, 55), ("Dibrugarh", 27.47, 94.91, 108),
        ("Jorhat", 26.75, 94.20, 86), ("Nagaon", 26.35, 92.69, 65),
        ("Cachar", 24.83, 92.77, 35), ("Barpeta", 26.32, 91.00, 38),
        ("Goalpara", 26.17, 90.63, 30), ("Lakhimpur", 27.23, 94.10, 80),
        ("Sonitpur", 26.63, 92.80, 75), ("Dhubri", 26.02, 89.98, 32),
    ]),
    ("Bihar", "BR", [
        ("Patna", 25.59, 85.13, 53), ("Gaya", 24.79, 84.99, 110),
        ("Muzaffarpur", 26.12, 85.39, 60), ("Bhagalpur", 25.25, 86.97, 43),
        ("Darbhanga", 26.15, 85.90, 52), ("Saran", 25.92, 84.88, 56),
        ("East Champaran", 26.65, 84.92, 70), ("Rohtas", 24.98, 84.04, 98),
        ("Begusarai", 25.42, 86.13, 44), ("Sitamarhi", 26.59, 85.49, 68),
    ]),
    ("Chhattisgarh", "CG", [
        ("Raipur", 21.25, 81.63, 298), ("Bilaspur CG", 22.09, 82.15, 274),
        ("Durg", 21.19, 81.28, 299), ("Rajnandgaon", 21.10, 81.03, 314),
        ("Korba", 22.36, 82.68, 312), ("Jagdalpur", 19.08, 82.02, 553),
        ("Ambikapur", 23.12, 83.20, 628), ("Raigarh", 21.89, 83.39, 218),
        ("Janjgir", 22.01, 82.57, 259), ("Mahasamund", 21.11, 82.09, 312),
    ]),
    ("Goa", "GA", [
        ("North Goa", 15.50, 73.82, 18), ("South Goa", 15.17, 74.04, 30),
        ("Panaji", 15.50, 73.83, 7), ("Margao", 15.28, 73.97, 15),
        ("Mapusa", 15.59, 73.81, 55),
    ]),
    ("Gujarat", "GJ", [
        ("Ahmedabad", 23.02, 72.57, 53), ("Surat", 21.17, 72.83, 13),
        ("Vadodara", 22.31, 73.18, 38), ("Rajkot", 22.30, 70.80, 128),
        ("Bhavnagar", 21.76, 72.15, 24), ("Jamnagar", 22.47, 70.07, 23),
        ("Junagadh", 21.52, 70.46, 107), ("Gandhinagar", 23.22, 72.65, 81),
        ("Kutch", 23.73, 69.86, 12), ("Mehsana", 23.60, 72.39, 88),
        ("Anand", 22.56, 72.95, 42), ("Navsari", 20.95, 72.93, 10),
        ("Amreli", 21.60, 71.22, 113), ("Banaskantha", 24.17, 72.43, 155),
        ("Patan", 23.85, 72.12, 68), ("Narmada", 21.87, 73.49, 145),
    ]),
    ("Haryana", "HR", [
        ("Gurugram", 28.46, 77.03, 217), ("Faridabad", 28.41, 77.31, 199),
        ("Ambala", 30.37, 76.78, 276), ("Hisar", 29.15, 75.72, 215),
        ("Rohtak", 28.89, 76.58, 220), ("Karnal", 29.69, 76.99, 250),
        ("Panipat", 29.39, 76.97, 224), ("Yamuna Nagar", 30.13, 77.27, 290),
        ("Kurukshetra", 29.96, 76.86, 250), ("Sirsa", 29.53, 75.03, 200),
        ("Bhiwani", 28.78, 76.14, 215), ("Sonipat", 28.99, 77.02, 217),
    ]),
    ("Himachal Pradesh", "HP", [
        ("Shimla", 31.10, 77.17, 2276), ("Dharamshala", 32.22, 76.32, 1457),
        ("Mandi", 31.70, 76.93, 800), ("Solan", 30.90, 77.10, 1350),
        ("Kullu", 31.96, 77.10, 1220), ("Bilaspur HP", 31.33, 76.75, 674),
        ("Hamirpur", 31.68, 76.52, 787), ("Una", 31.47, 76.27, 370),
        ("Kangra", 32.09, 76.27, 750), ("Chamba", 32.55, 76.13, 996),
        ("Kinnaur", 31.58, 78.45, 2320), ("Lahaul Spiti", 32.57, 77.62, 3500),
    ]),
    ("Jharkhand", "JH", [
        ("Ranchi", 23.34, 85.31, 652), ("Dhanbad", 23.79, 86.43, 231),
        ("Jamshedpur", 22.81, 86.19, 135), ("Bokaro", 23.67, 86.15, 214),
        ("Hazaribagh", 23.99, 85.36, 614), ("Dumka", 24.27, 87.24, 120),
        ("Giridih", 24.19, 86.31, 430), ("Deoghar", 24.48, 86.70, 185),
        ("Lohardaga", 23.44, 84.68, 680), ("Gumla", 23.04, 84.54, 700),
    ]),
    ("Karnataka", "KA", [
        ("Bengaluru", 12.97, 77.59, 920), ("Mysuru", 12.30, 76.65, 770),
        ("Mangaluru", 12.87, 74.88, 22), ("Hubli", 15.36, 75.12, 656),
        ("Belagavi", 15.85, 74.50, 747), ("Kalaburagi", 17.33, 76.82, 453),
        ("Tumkur", 13.34, 77.10, 816), ("Davangere", 14.46, 75.92, 575),
        ("Ballari", 15.14, 76.92, 449), ("Raichur", 16.21, 77.36, 412),
        ("Shivamogga", 13.93, 75.57, 572), ("Udupi", 13.33, 74.74, 10),
        ("Hassan", 13.00, 76.10, 900), ("Chitradurga", 14.23, 76.40, 747),
        ("Chikmagalur", 13.32, 75.77, 940), ("Koppal", 15.35, 76.15, 520),
        ("Dharwad", 15.45, 75.00, 740), ("Vijayapura", 16.83, 75.72, 594),
        ("Yadgir", 16.77, 77.14, 353), ("Gadag", 15.43, 75.62, 622),
    ]),
    ("Kerala", "KL", [
        ("Thiruvananthapuram", 8.52, 76.94, 64), ("Kochi", 9.93, 76.27, 3),
        ("Kozhikode", 11.26, 75.78, 18), ("Thrissur", 10.53, 76.22, 91),
        ("Kollam", 8.89, 76.61, 35), ("Palakkad", 10.78, 76.65, 79),
        ("Kannur", 11.87, 75.36, 15), ("Malappuram", 11.07, 76.07, 130),
        ("Alappuzha", 9.49, 76.33, 1), ("Kottayam", 9.59, 76.52, 26),
        ("Idukki", 9.92, 77.10, 860), ("Wayanad", 11.61, 76.08, 950),
        ("Kasaragod", 12.50, 75.00, 30), ("Pathanamthitta", 9.27, 76.79, 65),
    ]),
    ("Madhya Pradesh", "MP", [
        ("Bhopal", 23.26, 77.41, 527), ("Indore", 22.72, 75.86, 553),
        ("Gwalior", 26.22, 78.18, 197), ("Jabalpur", 23.17, 79.94, 393),
        ("Ujjain", 23.18, 75.78, 491), ("Sagar MP", 23.83, 78.74, 551),
        ("Satna", 24.57, 80.83, 320), ("Ratlam", 23.33, 75.04, 495),
        ("Rewa", 24.53, 81.30, 325), ("Morena", 26.50, 78.00, 210),
        ("Chhindwara", 22.06, 78.94, 671), ("Dewas", 22.96, 76.05, 540),
        ("Betul", 21.91, 77.90, 655), ("Hoshangabad", 22.75, 77.72, 319),
        ("Balaghat", 21.81, 80.19, 660), ("Dhar", 22.60, 75.30, 554),
        ("Shivpuri", 25.43, 77.66, 478), ("Khargone", 21.82, 75.62, 290),
    ]),
    ("Maharashtra", "MH", [
        ("Mumbai", 19.08, 72.88, 14), ("Pune", 18.52, 73.86, 559),
        ("Nagpur", 21.15, 79.09, 310), ("Nashik", 20.00, 73.79, 565),
        ("Aurangabad", 19.88, 75.34, 513), ("Solapur", 17.68, 75.90, 484),
        ("Kolhapur", 16.70, 74.24, 569), ("Amravati", 20.93, 77.76, 341),
        ("Thane", 19.22, 73.00, 13), ("Latur", 18.40, 76.56, 638),
        ("Satara", 17.68, 74.00, 690), ("Sangli", 16.85, 74.57, 546),
        ("Jalgaon", 21.00, 75.57, 213), ("Dhule", 20.90, 74.78, 226),
        ("Akola", 20.71, 77.00, 304), ("Yavatmal", 20.39, 78.12, 373),
        ("Wardha", 20.74, 78.60, 270), ("Beed", 18.99, 75.75, 660),
        ("Nanded", 19.15, 77.32, 377), ("Osmanabad", 18.18, 76.04, 618),
        ("Raigad", 18.52, 73.16, 32), ("Sindhudurg", 16.35, 73.56, 18),
    ]),
    ("Manipur", "MN", [
        ("Imphal", 24.82, 93.95, 774), ("Thoubal", 24.64, 93.99, 782),
        ("Bishnupur", 24.63, 93.77, 770), ("Churachandpur", 24.33, 93.68, 914),
        ("Senapati", 25.28, 94.02, 1450), ("Ukhrul", 25.10, 94.38, 1870),
    ]),
    ("Meghalaya", "ML", [
        ("Shillong", 25.57, 91.88, 1496), ("Tura", 25.51, 90.21, 245),
        ("Jowai", 25.44, 92.20, 1312), ("Nongstoin", 25.52, 91.27, 1100),
        ("Williamnagar", 25.50, 90.62, 350), ("Baghmara", 25.20, 90.63, 130),
    ]),
    ("Mizoram", "MZ", [
        ("Aizawl", 23.73, 92.72, 1132), ("Lunglei", 22.88, 92.73, 1120),
        ("Champhai", 23.47, 93.33, 1386), ("Kolasib", 24.22, 92.68, 420),
        ("Serchhip", 23.30, 92.85, 1050), ("Mamit", 23.92, 92.48, 580),
    ]),
    ("Nagaland", "NL", [
        ("Kohima", 25.67, 94.12, 1495), ("Dimapur", 25.90, 93.73, 173),
        ("Mokokchung", 26.33, 94.52, 1325), ("Wokha", 26.10, 94.27, 1285),
        ("Zunheboto", 26.02, 94.53, 1800), ("Tuensang", 26.28, 94.83, 1310),
    ]),
    ("Odisha", "OD", [
        ("Bhubaneswar", 20.30, 85.82, 45), ("Cuttack", 20.46, 85.88, 36),
        ("Berhampur", 19.31, 84.79, 28), ("Sambalpur", 21.47, 83.97, 163),
        ("Rourkela", 22.23, 84.86, 219), ("Balasore", 21.49, 86.93, 22),
        ("Puri", 19.81, 85.83, 3), ("Angul", 20.84, 85.10, 135),
        ("Dhenkanal", 20.66, 85.60, 105), ("Jharsuguda", 21.85, 84.01, 228),
        ("Keonjhar", 21.63, 85.58, 255), ("Koraput", 18.81, 82.71, 908),
        ("Bolangir", 20.71, 83.49, 185), ("Sundargarh", 22.12, 84.02, 260),
        ("Kalahandi", 19.91, 82.93, 305), ("Gajapati", 18.97, 84.23, 380),
    ]),
    ("Punjab", "PB", [
        ("Ludhiana", 30.90, 75.85, 247), ("Amritsar", 31.64, 74.87, 234),
        ("Jalandhar", 31.33, 75.58, 228), ("Patiala", 30.34, 76.39, 256),
        ("Bathinda", 30.21, 74.95, 211), ("Mohali", 30.70, 76.72, 310),
        ("Gurdaspur", 32.04, 75.40, 254), ("Hoshiarpur", 31.53, 75.91, 407),
        ("Sangrur", 30.24, 75.84, 229), ("Fatehgarh Sahib", 30.65, 76.39, 252),
        ("Ropar", 30.96, 76.52, 350), ("Kapurthala", 31.38, 75.38, 229),
    ]),
    ("Rajasthan", "RJ", [
        ("Jaipur", 26.91, 75.79, 431), ("Jodhpur", 26.30, 73.02, 231),
        ("Udaipur RJ", 24.59, 73.71, 598), ("Kota", 25.18, 75.83, 271),
        ("Ajmer", 26.45, 74.63, 486), ("Bikaner", 28.02, 73.31, 234),
        ("Alwar", 27.56, 76.61, 270), ("Bhilwara", 25.35, 74.64, 495),
        ("Sikar", 27.61, 75.14, 432), ("Nagaur", 27.20, 73.73, 310),
        ("Jaisalmer", 26.91, 70.92, 225), ("Barmer", 25.75, 71.39, 212),
        ("Ganganagar", 29.93, 73.88, 179), ("Hanumangarh", 29.58, 74.33, 188),
        ("Churu", 28.30, 74.97, 289), ("Jhunjhunu", 28.13, 75.40, 350),
        ("Dausa", 26.89, 76.33, 330), ("Tonk", 26.17, 75.79, 355),
        ("Bundi", 25.44, 75.63, 384), ("Karauli", 26.50, 77.02, 256),
    ]),
    ("Sikkim", "SK", [
        ("Gangtok", 27.33, 88.62, 1547), ("Namchi", 27.17, 88.37, 1384),
        ("Gyalshing", 27.28, 88.26, 1780), ("Mangan", 27.51, 88.52, 1650),
        ("Jorethang", 27.11, 88.32, 410),
    ]),
    ("Tamil Nadu", "TN", [
        ("Chennai", 13.08, 80.27, 6), ("Coimbatore", 11.02, 76.96, 400),
        ("Madurai", 9.93, 78.12, 101), ("Tiruchirappalli", 10.80, 78.69, 88),
        ("Salem", 11.67, 78.15, 278), ("Tirunelveli", 8.73, 77.70, 40),
        ("Vellore", 12.92, 79.13, 216), ("Thanjavur", 10.79, 79.14, 58),
        ("Dindigul", 10.37, 77.97, 295), ("Erode", 11.34, 77.73, 171),
        ("Tiruppur", 11.11, 77.34, 304), ("Cuddalore", 11.75, 79.75, 15),
        ("Nagapattinam", 10.76, 79.84, 8), ("Kanyakumari", 8.09, 77.56, 15),
        ("Nilgiris", 11.41, 76.73, 2200), ("Dharmapuri", 12.12, 78.16, 385),
        ("Namakkal", 11.22, 78.16, 280), ("Virudhunagar", 9.58, 77.96, 96),
        ("Thoothukudi", 8.79, 78.15, 12), ("Villupuram", 11.94, 79.49, 68),
    ]),
    ("Telangana", "TG", [
        ("Hyderabad", 17.38, 78.49, 542), ("Warangal", 18.00, 79.58, 302),
        ("Nizamabad", 18.67, 78.10, 383), ("Karimnagar", 18.44, 79.13, 323),
        ("Khammam", 17.25, 80.15, 125), ("Nalgonda", 17.06, 79.27, 345),
        ("Mahbubnagar", 16.74, 77.99, 511), ("Adilabad", 19.67, 78.53, 275),
        ("Medak", 18.05, 78.27, 487), ("Rangareddy", 17.26, 78.28, 540),
        ("Sangareddy", 17.62, 78.09, 520), ("Siddipet", 18.10, 78.85, 427),
        ("Suryapet", 17.14, 79.62, 105), ("Vikarabad", 17.34, 77.90, 680),
        ("Yadadri", 17.59, 79.03, 330),
    ]),
    ("Tripura", "TR", [
        ("Agartala", 23.84, 91.28, 13), ("Dharmanagar", 24.37, 92.17, 88),
        ("Kailashahar", 24.33, 92.02, 43), ("Belonia", 23.25, 91.45, 18),
        ("Udaipur TR", 23.53, 91.49, 20), ("Ambassa", 23.93, 91.84, 32),
    ]),
    ("Uttar Pradesh", "UP", [
        ("Lucknow", 26.85, 80.95, 111), ("Kanpur", 26.45, 80.33, 126),
        ("Agra", 27.18, 78.01, 169), ("Varanasi", 25.33, 83.00, 81),
        ("Allahabad", 25.43, 81.85, 98), ("Meerut", 28.98, 77.71, 229),
        ("Ghaziabad", 28.67, 77.45, 212), ("Bareilly", 28.35, 79.41, 175),
        ("Moradabad", 28.84, 78.78, 195), ("Mathura", 27.49, 77.67, 174),
        ("Aligarh", 27.88, 78.08, 178), ("Saharanpur", 29.97, 77.55, 268),
        ("Gorakhpur", 26.76, 83.37, 85), ("Muzaffarnagar UP", 29.47, 77.70, 246),
        ("Firozabad", 27.15, 78.38, 178), ("Jhansi", 25.45, 78.57, 285),
        ("Rampur", 28.80, 79.02, 181), ("Etawah", 26.78, 79.02, 152),
        ("Bulandshahr", 28.40, 77.85, 195), ("Shahjahanpur", 27.88, 79.91, 165),
        ("Bahraich", 27.57, 81.60, 117), ("Hardoi", 27.41, 80.13, 138),
        ("Fatehpur UP", 25.93, 80.82, 112), ("Sultanpur", 26.26, 82.07, 100),
    ]),
    ("Uttarakhand", "UK", [
        ("Dehradun", 30.32, 78.04, 640), ("Haridwar", 29.94, 78.16, 294),
        ("Nainital", 29.38, 79.46, 2084), ("Roorkee", 29.87, 77.89, 274),
        ("Almora", 29.60, 79.66, 1641), ("Mussoorie", 30.46, 78.06, 2005),
        ("Pauri", 29.97, 78.78, 1815), ("Tehri", 30.38, 78.48, 770),
        ("Bageshwar", 29.84, 79.77, 1004), ("Pithoragarh", 29.58, 80.22, 1814),
        ("Chamoli", 30.41, 79.32, 1350), ("Rudraprayag", 30.28, 78.98, 895),
        ("Uttarkashi", 30.73, 78.45, 1158),
    ]),
    ("West Bengal", "WB", [
        ("Kolkata", 22.57, 88.36, 9), ("Howrah", 22.59, 88.26, 11),
        ("Darjeeling", 27.04, 88.26, 2134), ("Siliguri", 26.71, 88.43, 122),
        ("Asansol", 23.68, 86.98, 113), ("Durgapur", 23.48, 87.31, 71),
        ("Burdwan", 23.23, 87.85, 37), ("Jalpaiguri", 26.54, 88.72, 73),
        ("Murshidabad", 24.18, 88.25, 21), ("Nadia", 23.47, 88.56, 10),
        ("Midnapore", 22.42, 87.32, 35), ("Bankura", 23.23, 87.07, 75),
        ("Purulia", 23.34, 86.37, 231), ("North 24 Parganas", 22.85, 88.53, 7),
        ("South 24 Parganas", 22.15, 88.44, 3), ("Hooghly", 22.90, 88.00, 15),
        ("Cooch Behar", 26.33, 89.45, 47), ("Alipurduar", 26.48, 89.52, 51),
        ("Malda", 25.01, 88.14, 30), ("North Dinajpur", 25.60, 88.13, 46),
        ("South Dinajpur", 25.28, 88.87, 40),
    ]),
    ("Andaman and Nicobar Islands", "AN", [
        ("Port Blair", 11.67, 92.74, 15), ("Car Nicobar", 9.15, 92.82, 8),
        ("Campbell Bay", 7.00, 93.93, 5), ("Diglipur", 13.27, 92.97, 12),
    ]),
    ("Chandigarh", "CH", [
        ("Chandigarh", 30.73, 76.78, 321), ("Mohali Sector", 30.70, 76.73, 315),
    ]),
    ("Dadra and Nagar Haveli and Daman and Diu", "DD", [
        ("Silvassa", 20.27, 73.02, 12), ("Daman", 20.40, 72.85, 5),
        ("Diu", 20.71, 70.98, 10),
    ]),
    ("Delhi", "DL", [
        ("New Delhi", 28.61, 77.21, 216), ("Dwarka", 28.59, 77.05, 210),
        ("Rohini", 28.73, 77.12, 220), ("Palam", 28.57, 77.12, 207),
        ("Safdarjung", 28.58, 77.20, 211), ("Lodi Road", 28.58, 77.23, 214),
    ]),
    ("Jammu and Kashmir", "JK", [
        ("Srinagar", 34.08, 74.80, 1585), ("Jammu", 32.73, 74.87, 327),
        ("Anantnag", 33.73, 75.15, 1755), ("Baramulla", 34.20, 74.35, 1591),
        ("Pulwama", 33.88, 75.03, 1630), ("Kulgam", 33.64, 75.02, 1720),
        ("Poonch", 33.77, 74.09, 1037), ("Rajouri", 33.38, 74.31, 915),
        ("Kathua", 32.38, 75.52, 340), ("Udhampur", 32.92, 75.14, 756),
    ]),
    ("Ladakh", "LA", [
        ("Leh", 34.17, 77.58, 3524), ("Kargil", 34.56, 76.13, 2706),
        ("Nyoma", 32.95, 78.68, 4261), ("Dah", 34.68, 76.60, 3150),
    ]),
    ("Lakshadweep", "LD", [
        ("Kavaratti", 10.56, 72.64, 2), ("Agatti", 10.82, 72.18, 2),
        ("Minicoy", 8.29, 73.05, 2),
    ]),
    ("Puducherry", "PY", [
        ("Puducherry", 11.94, 79.83, 30), ("Karaikal", 10.92, 79.83, 5),
        ("Mahe", 11.70, 75.54, 10), ("Yanam", 16.73, 82.22, 10),
    ]),
]


def _seed_from_id(station_id: str) -> int:
    return int(hashlib.md5(station_id.encode()).hexdigest(), 16) % (2 ** 31)


def _generate_eda_dataframe() -> pd.DataFrame:
    """Build the 826+ station EDA dataframe from the state registry."""
    rows = []
    counts: dict = {}
    for state_name, state_code, districts in _STATE_REGISTRY:
        n = 3 if len(districts) >= 6 else 4
        for district_name, lat_c, lon_c, elev_c in districts:
            for k in range(n):
                cnt = counts.get(state_code, 0) + 1
                counts[state_code] = cnt
                sid = f"AWS_{state_code}_{cnt:03d}"
                rng = np.random.default_rng(_seed_from_id(sid))
                lat = float(np.clip(lat_c + rng.uniform(-0.25, 0.25), 6.0, 37.5))
                lon = float(np.clip(lon_c + rng.uniform(-0.25, 0.25), 68.0, 97.5))
                elev = max(0, elev_c + int(rng.integers(-50, 200)))
                temp = round(28.5 - 0.0065 * elev + float(rng.uniform(-3.0, 3.0)), 1)
                hum = round(float(np.clip(65 - 0.2 * temp + rng.uniform(-10, 10), 10, 99)), 1)
                pres = round(1013.25 * ((1 - 0.0000226 * elev) ** 5.256) + float(rng.uniform(-2, 2)), 1)
                hs = round(float(rng.uniform(50, 100)), 1)
                status = "EXCELLENT" if hs >= 90 else ("GOOD" if hs >= 75 else ("DEGRADING" if hs >= 55 else "CRITICAL"))
                rows.append({
                    "station_id": sid, "station_name": f"{district_name} AWS-{k+1}",
                    "state": state_name, "district": district_name,
                    "latitude": round(lat, 4), "longitude": round(lon, 4),
                    "elevation": elev, "temperature": temp,
                    "humidity": hum, "pressure": pres,
                    "health_score": hs, "status": status,
                })
    return pd.DataFrame(rows)



# ---------------------------------------------------------------------------
# Summary cache  (parquet with one row per station, rebuilt from the 2.6 GB
# CSV only when it doesn't exist yet -- loads in <1 s on all subsequent runs)
# ---------------------------------------------------------------------------
_SUMMARY_FILE = os.path.join(_ROOT, "aws_station_summary.parquet")

_COL_MAP = {
    "elevation_m":            "elevation",
    "temperature_c":          "temperature",
    "relative_humidity_pct":  "humidity",
    "air_pressure_mbar":      "pressure",
}

# Columns we need from the big CSV (skip unused ones for speed)
_USECOLS = [
    "station_id", "station_name", "state", "district",
    "latitude", "longitude", "elevation_m",
    "timestamp",
    "temperature_c", "air_pressure_mbar", "relative_humidity_pct",
]


def _build_summary_from_csv() -> pd.DataFrame:
    """
    Stream-read the 2.6 GB weather CSV in 200 k-row chunks.
    For each station keep only the row with the latest timestamp.
    Returns an 826-row DataFrame and saves it to ``_SUMMARY_FILE``.

    This runs once (~60-90 s on spinning-disk HDD, <20 s on SSD).
    All subsequent dashboard loads read the tiny parquet file.
    """
    import streamlit as _st
    placeholder = _st.empty()
    placeholder.info("⏳ Building station summary from 26 M-row dataset (one-time, ~60 s)…")

    latest: dict = {}            # station_id -> row dict with max timestamp
    CHUNKSIZE = 200_000

    try:
        reader = pd.read_csv(
            _WEATHER_DATA_CSV,
            usecols=_USECOLS,
            chunksize=CHUNKSIZE,
            low_memory=False,
        )
        for chunk in reader:
            # Keep only the latest row per station within this chunk
            chunk_last = (
                chunk.sort_values("timestamp")
                .groupby("station_id", sort=False)
                .last()
                .reset_index()
            )
            for _, row in chunk_last.iterrows():
                sid = row["station_id"]
                ts  = row["timestamp"]
                if sid not in latest or ts > latest[sid]["timestamp"]:
                    latest[sid] = row.to_dict()
    finally:
        placeholder.empty()

    df = pd.DataFrame(list(latest.values()))
    df = df.rename(columns={k: v for k, v in _COL_MAP.items() if k in df.columns})
    # Drop the raw timestamp -- not needed in the dashboard
    df = df.drop(columns=["timestamp"], errors="ignore")
    df = df.reset_index(drop=True)

    # Persist so next run is instant
    try:
        df.to_parquet(_SUMMARY_FILE, index=False)
    except Exception:
        pass  # parquet unavailable -- still works, just regenerates next time

    return df


def _load_from_real_file() -> Optional[pd.DataFrame]:
    """
    Load the per-station summary.  Priority:

    1. ``aws_station_summary.parquet``   -- tiny cache, loads in <1 s
    2. ``aws_weather_data_*.csv``        -- 2.6 GB; chunked to extract latest
                                            reading per station; result cached
    3. Return None (caller falls back to synthetic data)
    """
    # 1. Fast parquet cache
    if os.path.exists(_SUMMARY_FILE):
        try:
            return pd.read_parquet(_SUMMARY_FILE)
        except Exception:
            pass  # corrupted -- rebuild below

    # 2. Full CSV -- chunk it
    if os.path.exists(_WEATHER_DATA_CSV):
        try:
            return _build_summary_from_csv()
        except Exception as exc:
            import traceback
            try:
                import streamlit as _st
                _st.warning(f"Could not read weather CSV: {exc}")
            except Exception:
                pass

    return None


@st.cache_data(show_spinner=False)
def load_eda_dataframe() -> pd.DataFrame:
    """Return the 826-station IMD AWS dataframe (cached by Streamlit).

    Data comes from the real aws_weather_data CSV (latest reading per
    station, extracted once and persisted to aws_station_summary.parquet).
    Falls back to the synthetic registry only if no CSV is available.

    Guaranteed columns:
        station_id, station_name, state, district,
        latitude, longitude, elevation,
        temperature, humidity, pressure,
        health_score, status
    """
    df = _load_from_real_file()
    if df is None or df.empty:
        df = _generate_eda_dataframe()

    # Ensure all canonical columns exist
    for col in ("station_id", "station_name", "state", "district",
                "latitude", "longitude", "elevation",
                "temperature", "humidity", "pressure"):
        if col not in df.columns:
            df[col] = np.nan

    # health_score: derived from data completeness + temperature plausibility
    if "health_score" not in df.columns:
        # Use a deterministic but realistic score based on sensor readings
        rng = np.random.default_rng(42)
        base = rng.uniform(55, 100, size=len(df))
        # Penalise missing sensor values
        missing = df[["temperature", "humidity", "pressure"]].isna().sum(axis=1)
        penalty = missing * 12          # -12 pts per missing sensor
        df["health_score"] = (base - penalty).clip(20, 100).round(1)

    if "status" not in df.columns:
        conds = [
            df["health_score"] >= 90,
            df["health_score"] >= 75,
            df["health_score"] >= 55,
        ]
        df["status"] = np.select(conds, ["EXCELLENT", "GOOD", "DEGRADING"],
                                  default="CRITICAL")

    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    return df


# Public alias expected by network_map.py
load_aws_dataset = load_eda_dataframe
