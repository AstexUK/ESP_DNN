#!/usr/bin/perl

system("rm -r cif/*");

my $file_open = 0;

open(CIF,"<$ARGV[0]");

while (my $line = <CIF>) {

  if ($line =~ /^data_/) {

    my $code = substr($line,5);

    chomp($code);

    if ($file_open) {

      close(HETCIF);

      $file_open = 0;
    }

    my $dir = (length($code) > 1) ? "cif/".substr($code,0,2) : "cif/".substr($code,0,1);

    my $filename = "$code.cif";

    if (!-e $dir) {

      system("mkdir -p $dir");
    }

    print("file '$dir/$filename' already exists!\n") if (-e "$dir/$filename");

    $file_open = open(HETCIF,">$dir/$filename");
  }

  if ($file_open) {

    print HETCIF $line;

  } else {

    print "skipping line: $line";
  }
}

close(CIF);
close(HETCIF) if ($file_open);

exit;
