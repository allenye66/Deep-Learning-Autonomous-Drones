//
//  ViewController.swift
//  SynopsysApp
//
//  Created by Allen on 3/8/20.
//  Copyright © 2020 Allen. All rights reserved.
//

import UIKit
import CoreLocation
class ViewController: UIViewController, CLLocationManagerDelegate {
    @IBOutlet weak var getHelp: UIButton!
    let locationManager:CLLocationManager = CLLocationManager()
    override func viewDidLoad() {
        super.viewDidLoad()
        getHelp.layer.cornerRadius = 40;
        getHelp.clipsToBounds = true
        
        print("trying to get location")
        // Do any additional setup after loading the view.
        locationManager.delegate = self
        locationManager.requestWhenInUseAuthorization()
        locationManager.startUpdatingLocation()

    }
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]){
        for currentLocation in locations{
       //     print("\(index): \(currentLocation)")
        }
    }


}

