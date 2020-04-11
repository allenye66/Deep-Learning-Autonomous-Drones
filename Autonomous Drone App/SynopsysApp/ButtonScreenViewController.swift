//
//  ButtonScreenViewController.swift
//  SynopsysApp
//
//  Created by Allen on 3/10/20.
//  Copyright © 2020 Allen. All rights reserved.
//

import UIKit

class ButtonScreenViewController: UIViewController {

    
    @IBOutlet weak var getHelpButton: UIButton!
    override func viewDidLoad() {
        super.viewDidLoad()
        //getHelpButton.layer.cornerRadius = 20;
        getHelpButton.clipsToBounds = true

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
