(local vips  (require "vips"))

(fn say-hi []
  (print "hi"))

(fn wfc [config image]
  (print config))


(fn load-image [path opts?]
  (vips.Image.new_from_file path opts?))

(fn read-pixel [image position]
  (let [[x y] position
        [r g b] (image x y)]
    {:r r :g g :b b}))


(local water-path "samples/Water.png")
(local image (load-image water-path))
(local [x y z] (image 1 2))
(print x)


(let [image (load-image water-path)]
  (print (read-pixel image [1 2])))
