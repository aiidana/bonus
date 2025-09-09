-- CREATE DATABASE transport_db;
USE transport_db;
-- CREATE TABLE Companies (
--     company_id INT PRIMARY KEY,
--     company_name VARCHAR(255)
-- );

-- CREATE TABLE Vehicles (
--     vehicle_id INT PRIMARY KEY,
--     vehicle_type VARCHAR(255),
--     company_id INT,
--     FOREIGN KEY (company_id) REFERENCES Companies(company_id)
-- );

-- CREATE TABLE Cities (
--     city_id INT PRIMARY KEY,
--     city_name VARCHAR(255),
--     country_id INT
-- );

-- CREATE TABLE Routes (
--     route_id INT PRIMARY KEY,
--     city_id INT,
--     FOREIGN KEY (city_id) REFERENCES Cities(city_id)
-- );

-- CREATE TABLE Transport_Types (
--     transport_id INT PRIMARY KEY,
--     transport_name VARCHAR(255),
--     average_speed DECIMAL
-- );
-- INSERT INTO Companies (company_id, company_name) VALUES
-- (1, 'Alpha Transport'),
-- (2, 'Beta Logistics'),
-- (3, 'Gamma Movers'),
-- (4, 'Delta Cargo'),
-- (5, 'Epsilon Express');

-- INSERT INTO Vehicles (vehicle_id, vehicle_type, company_id) VALUES
-- (1, 'Truck', 1),
-- (2, 'Bus', 2),
-- (3, 'Van', 3),
-- (4, 'Car', 4),
-- (5, 'Motorbike', 5);
-- INSERT INTO Cities (city_id, city_name, country_id) VALUES
-- (1, 'New York', 1),
-- (2, 'Los Angeles', 1),
-- (3, 'Chicago', 1),
-- (4, 'Almaty', 2),
-- (5, 'Astana', 2);

-- INSERT INTO Routes (route_id, city_id) VALUES
-- (1, 1),
-- (2, 2),
-- (3, 3),
-- (4, 4),
-- (5, 5);

-- INSERT INTO Transport_Types (transport_id, transport_name, average_speed) VALUES
-- (1, 'Train', 80),
-- (2, 'Bus', 60),
-- (3, 'Airplane', 600),
-- (4, 'Ship', 40),
-- (5, 'Bicycle', 20);


-- SELECT c.company_name, COUNT(v.vehicle_id) AS vehicle_count
-- FROM Companies c
-- LEFT JOIN Vehicles v ON c.company_id = v.company_id
-- GROUP BY c.company_name;

-- SELECT * FROM Companies
-- WHERE company_name LIKE 'A%';

-- SELECT * FROM Cities
-- WHERE country_id BETWEEN 3 AND 10;

-- SELECT vehicle_id, vehicle_type FROM Vehicles;

-- SELECT r.route_id, c.city_name 
-- FROM Routes r
-- JOIN Cities c ON r.city_id = c.city_id;

-- SELECT * FROM Transport_Types;
-- UPDATE Transport_Types
-- SET average_speed = average_speed + 10
-- WHERE average_speed < 100;
-- SELECT * FROM Transport_Types;

-- SELECT city_name, country_id 
-- FROM Cities;

-- SELECT * FROM Transport_Types;











