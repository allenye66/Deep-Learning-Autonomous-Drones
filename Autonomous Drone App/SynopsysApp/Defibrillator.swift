//
//  Defibrillator.swift
//  SynopsysApp
//
//  Created by Allen on 3/10/20.
//  Copyright © 2020 Allen. All rights reserved.
//

import UIKit
import Firebase
import CoreLocation

class Defibrillator: UIViewController, CLLocationManagerDelegate {
    
    let locationManager:CLLocationManager = CLLocationManager()
    var coordinates:CLLocation!

    override func viewDidLoad() {
        super.viewDidLoad()
        print("cardiac arrest")
        locationManager.delegate = self
       // locationManager.requestWhenInUseAuthorization()
        locationManager.startUpdatingLocation()
        coordinates = locationManager.location
        print(coordinates)
        let ref = Database.database().reference()
        
        ref.child("Supplies needed: face masks").setValue(["Latitude": locationManager.location!.coordinate.latitude, "Longitude": locationManager.location!.coordinate.longitude])
        // Do any additional setup after loading the view.
    }
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]){
        //for currentLocation in locations{
         //  print("\(index): \(currentLocation)")
        //}
    }
    

    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destination.
        // Pass the selected object to the new view controller.
    }
    */

}
