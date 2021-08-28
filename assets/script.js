
function pause() {
    pywebview.api.pause()
}

function quit() {
    pywebview.api.quit()
}

function start() {
    pywebview.api.start()
}

function clear_user_data() {
    pywebview.api.clear_user_data()
}

function add_profile() {
    var name = document.getElementById("user-box").value;

    pywebview.api.add_profile(name);
}

function load_profiles() {

    const blocked_chars = ['[', ']', ',', /["']/g];
    pywebview.api.load_profiles().then(function(names) {

        // iterate through and remove all of blocked_chars from the names
        for (let i=0; i<blocked_chars.length; i++) {
            if (blocked_chars[i] == ",") {
                names = names.replace(blocked_chars[i], " ");
            } else {
                names = names.replace(blocked_chars[i], "");
            }
        }

        const name_array = names.split(" ");

        for (let i=0; i<name_array.length; i++) {
            var name = name_array[i];

            // capitalize first letter of name
            name = name.charAt(0).toUpperCase() + name.slice(1);

            // create and place new p element for each name
            const e = document.createElement('p');
            e.innerHTML = name;
            e.className = "profile-name";
            document.body.appendChild(e);
        }
    });
}
