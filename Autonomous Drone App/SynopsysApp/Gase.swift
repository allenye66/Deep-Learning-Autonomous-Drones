//
//  Gase.swift
//  
//
//  Created by Allen on 3/10/20.
//

import UIKit
import Firebase
class Gase: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()

        // Do any additional setup after loading the view.
        let ref = Database.database().reference()
        //let arr = [100, 100]
        ref.child("Cardiac Arrest").setValue("ooga")

        
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
