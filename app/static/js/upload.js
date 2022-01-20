$(document).ready(function () {
    var access_token = $('#a_token').text();
    $('#people').click();
    $('#ctn-preloader').css("display", "none");
    
    $('#file-picker').change(function () {
        $("#submit-file").click();
        $('#ctn-preloader').css("display", "flex");
        $('#ctn-preloader').addClass('loaded');
          $("#loading").fadeOut(500);
          // Una vez haya terminado el preloader aparezca el scroll
      
          if ($('#ctn-preloader').hasClass('loaded')) {
            // Es para que una vez que se haya ido el preloader se elimine toda la seccion preloader
            $('#preloader').delay(900).queue(function () {
              $(this).remove();
            });
          }
    });
});
