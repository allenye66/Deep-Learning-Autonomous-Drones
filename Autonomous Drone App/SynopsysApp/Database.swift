//
//  Database.swift
//  SynopsysApp
//
//  Created by Allen on 3/10/20.
//  Copyright © 2020 Allen. All rights reserved.
//

import UIKit
import Firebase
class ooga: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        var ref: DatabaseReference!

        ref = (Database.database() as AnyObject).reference()
        
        ref.child("Symptom").setValue("Cardiac Arrest")
        // Do any additional setup after loading the view.
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
