
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

function load_conversation(name) {
    var conversation_data;
    pywebview.api.load_conversation(name).then(function(parsed_convo) {
            conversation_data = parsed_convo;
            const e = document.createElement('p');
            e.innerHTML = conversation_data;
            document.body.appendChild(e);
        }
    )
}

function parse_imported_list(list) {

    const blocked_chars = ['[', ']', ',', /["']/g];
    // iterate through and remove all of blocked_chars from the names
    for (let i=0; i<blocked_chars.length; i++) {
        if (blocked_chars[i] == ",") {
            list = list.replace(blocked_chars[i], " ");
        } else {
            list = list.replace(blocked_chars[i], "");
        }
    }

    const element_array = list.split(" ");

    return element_array;

}

function load_profiles() {

    pywebview.api.load_profiles().then(function(names) {

        var name_array = parse_imported_list(names);

        for (let i=0; i<name_array.length; i++) {
            var name = name_array[i];

            // capitalize first letter of name
            var cap_name = name.charAt(0).toUpperCase() + name.slice(1);

            // create and place new p element for each name
            const e = document.createElement('p');
            e.innerHTML = cap_name;
            e.className = "profile-name";
            document.body.appendChild(e);

            // create and place new button element to show conversations for each name
            const b = document.createElement('button');
            b.innerHTML = "Conversations";
            b.className = "conversation-button";
            b.onclick = function() { load_conversation(name); };
            document.body.appendChild(b);
        }
    });

}
