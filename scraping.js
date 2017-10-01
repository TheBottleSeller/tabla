var scrapeAudio = function() {
  var oddRows = document.getElementsByClassName('odd');
  var evenRows = document.getElementsByClassName('even');

  if (oddRows.length !== 21 && evenRows.length !== 21) {
    console.log("INCORRECT NUMBER OF RECORDINGS");
    return;
  }

  var rows = [];
  for (var i = 0; i < 21; i++) {
    rows.push(oddRows[i]);
    rows.push(evenRows[i]);
  }

  rows.sort(function(row1, row2) {
    if (row1.id < row2.id) {
      return -1
    } else {
      return 1
    }
  })

  var rowToFilename = [
    "TF/TF_RLL_1.wav",
    "TF/TF_LLL_1.wav",
    "TF/TF_RML_1.wav",
    "TF/TF_LML_1.wav",
    "TF/TF_RUL_1.wav",
    "TF/TF_LUL_1.wav",
    "BS/BS_RLL_3.wav",
    "BS/BS_LLL_3.wav",
    "BS/BS_RML_3.wav",
    "BS/BS_LML_3.wav",
    "BS/BS_RUL_3.wav",
    "BS/BS_LUL_3.wav",
    "BS/BS_RLL_2.wav",
    "BS/BS_LLL_2.wav",
    "BS/BS_RML_2.wav",
    "BS/BS_LML_2.wav",
    "BS/BS_RUL_2.wav",
    "BS/BS_LUL_2.wav",
    "BS/BS_RLL_1.wav",
    "BS/BS_LLL_1.wav",
    "BS/BS_RML_1.wav",
    "BS/BS_LML_1.wav",
    "BS/BS_RUL_1.wav",
    "BS/BS_LUL_1.wav",
    "PS/PS_RLL_3.wav",
    "PS/PS_LLL_3.wav",
    "PS/PS_RML_3.wav",
    "PS/PS_LML_3.wav",
    "PS/PS_RUL_3.wav",
    "PS/PS_LUL_3.wav",
    "PS/PS_RLL_2.wav",
    "PS/PS_LLL_2.wav",
    "PS/PS_RML_2.wav",
    "PS/PS_LML_2.wav",
    "PS/PS_RUL_2.wav",
    "PS/PS_LUL_2.wav",
    "PS/PS_RLL_1.wav",
    "PS/PS_LLL_1.wav",
    "PS/PS_RML_1.wav",
    "PS/PS_LML_1.wav",
    "PS/PS_RUL_1.wav",
    "PS/PS_LUL_1.wav",
  ];

  var cmds = [
    "mkdir BS; mkdir PS; mkdir TF;"
  ];

  var generateCmd = function(i) {
    if (i === 42) {
      console.log(cmds)
      var script = cmds.join("\n") + "\n";
      script
      console.log(script);
      return;
    }
    var audioUrl = "https://dashboard.ekodevices.com/#/dashboard/recordings/" + rows[i].id;
    if (window.location.href !== audioUrl) {
      window.location.href = audioUrl;
      return setTimeout(function() { generateCmd(i) }, 3000)
    }
    var downloadElement = document.getElementById('download-sound-td')
    if (!downloadElement) {
      return setTimeout(function() { generateCmd(i) }, 3000)
    }
    if (!downloadElement.children[0].href) {
      return setTimeout(function() { generateCmd(i) }, 3000)
    }

    var url = "\"" + downloadElement.children[0].href + "\"";
    var filename = rowToFilename[i]
    var cmd = "sleep 1; curl -o " + filename + " " + url + ";"
    cmds.push(cmd)
    console.log(cmd)
    generateCmd(i + 1)
  }

  generateCmd(0);
}
